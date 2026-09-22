"""Stage 0: leakage-safe data construction (specification section 13).

Builds, from the raw competition files:

  dbf2_prepared/
    views/peaks_mz.npy       flat float64 store, full-precision m/z, no binning
    views/peaks_int.npy      flat float32 store, base-peak-normalised intensity
    views/spectra.parquet    one row per view, sorted so a molecule's views are contiguous
    molecules.parquet        one row per structure: smiles, formula, mass, scaffold, fold
    targets.parquet          BRICS presence and functional-group counts
    target_dictionary.csv    target definitions and fit prevalence
    oracle_test_key.csv      recovered test answer key (see the caution below)
    manifest.json            input digests and stage completion flags

CAUTION. Every test spectrum in the released CASMI 2026 data occurs as an exact
duplicate inside train.parquet, carrying its structure. ``build_oracle`` recovers
that mapping. It exists so the 400 test structures can be EXCLUDED from training
and used as an honest held-out benchmark. It must not be used to produce a
competition submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import Config, DataConfig
from . import chem

PEAK_CHUNK = 1 << 22


# --------------------------------------------------------------------------- utils
def _manifest_path(cfg: Config) -> Path:
    return cfg.paths.prepared / "manifest.json"


def load_manifest(cfg: Config) -> dict:
    p = _manifest_path(cfg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"stages": {}, "shards": []}


def save_manifest(cfg: Config, man: dict) -> None:
    p = _manifest_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(man, indent=1), encoding="utf-8")


def clean_peaks(mz, inten, precursor_mz: float, dc: DataConfig
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Peak cleaning. m/z is kept in float64 at full precision: no binning, no rounding."""
    mz = np.asarray(mz, dtype=np.float64)
    it = np.asarray(inten, dtype=np.float64)
    if mz.size == 0:
        return mz, it.astype(np.float32)
    ok = (np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz > 0)
          & (mz < precursor_mz + dc.precursor_window))
    mz, it = mz[ok], it[ok]
    if mz.size == 0:
        return mz, it.astype(np.float32)
    it = it / it.max()
    keep = it >= dc.rel_intensity_floor
    mz, it = mz[keep], it[keep]
    if dc.max_peaks and mz.size > dc.max_peaks:
        sel = np.argpartition(it, -dc.max_peaks)[-dc.max_peaks:]
        mz, it = mz[sel], it[sel]
    order = np.argsort(mz, kind="stable")
    return mz[order], it[order].astype(np.float32)


# ------------------------------------------------------- stage A: clean the spectra
_TRAIN_COLS = ["inchikey14", "normalized_smiles", "molecular_formula", "ingest_lib",
               "ionization_mode", "instrument_type", "adduct", "precursor_mz",
               "precursor_error_ppm", "ms2_mzs", "ms2_normalized_intensities",
               "collision_energy_ev", "collision_energy_orig_units"]
_TEST_COLS = ["molecule_id", "spectrum_id", "ionization_mode", "instrument_type",
              "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
              "collision_energy_ev", "collision_energy_orig_units"]


def _process_frame(df: pd.DataFrame, dc: DataConfig, is_train: bool, stats: dict):
    stats["in"] = stats.get("in", 0) + len(df)
    if is_train:
        bad = (df.precursor_mz > dc.max_precursor_mz) | (~np.isfinite(df.precursor_mz))
        bad_ppm = df.precursor_error_ppm.abs() > dc.max_precursor_ppm   # NaN kept
        stats["drop_precursor"] = stats.get("drop_precursor", 0) + int(bad.sum())
        stats["drop_ppm"] = stats.get("drop_ppm", 0) + int((bad_ppm & ~bad).sum())
        df = df[~(bad | bad_ppm.fillna(False))]

    peaks = [clean_peaks(m, i, p, dc) for m, i, p
             in zip(df.ms2_mzs, df.ms2_normalized_intensities, df.precursor_mz)]
    ce = [chem.harmonise_energy(e, u, p) for e, u, p
          in zip(df.collision_energy_ev, df.collision_energy_orig_units, df.precursor_mz)]

    meta = pd.DataFrame({
        "precursor_mz": df.precursor_mz.to_numpy(np.float64),
        "adduct": df.adduct.to_numpy(),
        "polarity": df.ionization_mode.to_numpy(),
        "instrument_family": [chem.instrument_family(x) for x in df.instrument_type],
        "ce_ev": np.array([c[0] for c in ce], dtype=np.float32),
        "ce_missing": np.array([c[1] for c in ce], dtype=np.int8),
        "n_peaks": np.array([len(p[0]) for p in peaks], dtype=np.int32),
    })
    if is_train:
        meta.insert(0, "inchikey14", df.inchikey14.to_numpy())
        meta["ingest_lib"] = df.ingest_lib.to_numpy()
        few = meta.n_peaks.to_numpy() < dc.min_peaks
        stats["drop_few_peaks"] = stats.get("drop_few_peaks", 0) + int(few.sum())
        keep = ~few
        meta = meta[keep].reset_index(drop=True)
        peaks = [p for p, k in zip(peaks, keep) if k]
    else:
        meta.insert(0, "molecule_id", df.molecule_id.to_numpy())
        meta.insert(1, "spectrum_id", df.spectrum_id.to_numpy())
    stats["out"] = stats.get("out", 0) + len(meta)
    return meta, peaks


class _FlatWriter:
    """Append-only flat peak store with a growing memmap."""

    def __init__(self, mz_path: Path, int_path: Path):
        mz_path.parent.mkdir(parents=True, exist_ok=True)
        self.fmz = open(mz_path, "wb")
        self.fint = open(int_path, "wb")
        self.offset = 0

    def append(self, mz: np.ndarray, it: np.ndarray) -> int:
        off = self.offset
        self.fmz.write(np.ascontiguousarray(mz, dtype=np.float64).tobytes())
        self.fint.write(np.ascontiguousarray(it, dtype=np.float32).tobytes())
        self.offset += mz.size
        return off

    def close(self):
        self.fmz.close()
        self.fint.close()


def stage_spectra(cfg: Config, deadline: float) -> bool:
    """Clean every spectrum and write the flat peak store. Resumable per row group."""
    man = load_manifest(cfg)
    vd = cfg.paths.views
    vd.mkdir(parents=True, exist_ok=True)

    # test set first: cheap, and never filtered
    if "test" not in man["stages"]:
        st: dict = {}
        df = pq.read_table(cfg.paths.test_parquet, columns=_TEST_COLS).to_pandas()
        meta, peaks = _process_frame(df, cfg.data, False, st)
        w = _FlatWriter(vd / "test_mz.npy.raw", vd / "test_int.npy.raw")
        meta["offset"] = [w.append(*p) for p in peaks]
        w.close()
        meta.to_parquet(vd / "test_spectra.parquet")
        man["stages"]["test"] = st
        save_manifest(cfg, man)
        print(f"[stage A] test: {st}", flush=True)

    f = pq.ParquetFile(cfg.paths.train_parquet)
    n_rg = f.metadata.num_row_groups
    if cfg.data.limit_row_groups:
        n_rg = min(n_rg, int(cfg.data.limit_row_groups))
    done = set(man.get("shards", []))
    st = man["stages"].get("train", {})
    for g in range(n_rg):
        if g in done:
            continue
        if time.time() > deadline:
            print("[stage A] budget exhausted; rerun to resume", flush=True)
            return False
        df = f.read_row_group(g, columns=_TRAIN_COLS).to_pandas()
        meta, peaks = _process_frame(df, cfg.data, True, st)
        w = _FlatWriter(vd / f"part_{g:02d}_mz.raw", vd / f"part_{g:02d}_int.raw")
        meta["offset"] = [w.append(*p) for p in peaks]
        w.close()
        meta["part"] = g
        meta.to_parquet(vd / f"part_{g:02d}.parquet")
        done.add(g)
        man["shards"] = sorted(done)
        man["stages"]["train"] = st
        save_manifest(cfg, man)
        print(f"[stage A] row group {g}/{n_rg-1}: {len(meta)} spectra", flush=True)
    return True


# ------------------------------------------- stage B: regroup so views are contiguous
def stage_regroup(cfg: Config, deadline: float) -> bool:
    man = load_manifest(cfg)
    if "regroup" in man["stages"]:
        return True
    vd = cfg.paths.views
    parts = sorted(vd.glob("part_*.parquet"))
    if not parts:
        raise RuntimeError("stage A has not produced any shards")

    meta = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    keys = np.sort(meta.inchikey14.unique())
    kmap = {k: i for i, k in enumerate(keys)}
    meta["mol_idx"] = meta.inchikey14.map(kmap).astype(np.int32)
    order = np.lexsort((meta.part.to_numpy(), meta.mol_idx.to_numpy()))
    meta = meta.iloc[order].reset_index(drop=True)

    total = int(meta.n_peaks.sum())
    mz_out = np.lib.format.open_memmap(vd / "peaks_mz.npy", mode="w+",
                                       dtype=np.float64, shape=(total,))
    int_out = np.lib.format.open_memmap(vd / "peaks_int.npy", mode="w+",
                                        dtype=np.float32, shape=(total,))
    src_mz = {g: np.memmap(vd / f"part_{g:02d}_mz.raw", dtype=np.float64, mode="r")
              for g in meta.part.unique()}
    src_int = {g: np.memmap(vd / f"part_{g:02d}_int.raw", dtype=np.float32, mode="r")
               for g in meta.part.unique()}

    new_off = np.empty(len(meta), dtype=np.int64)
    cur = 0
    for i, (part, off, n) in enumerate(zip(meta.part.to_numpy(),
                                           meta.offset.to_numpy(),
                                           meta.n_peaks.to_numpy())):
        mz_out[cur:cur + n] = src_mz[part][off:off + n]
        int_out[cur:cur + n] = src_int[part][off:off + n]
        new_off[i] = cur
        cur += n
    mz_out.flush(); int_out.flush()
    meta["offset"] = new_off
    meta.drop(columns=["part"]).to_parquet(vd / "spectra.parquet")

    # the test store needs the same .npy framing
    for tag, dt in (("mz", np.float64), ("int", np.float32)):
        raw = np.memmap(vd / f"test_{tag}.npy.raw", dtype=dt, mode="r")
        arr = np.lib.format.open_memmap(vd / f"test_peaks_{tag}.npy", mode="w+",
                                        dtype=dt, shape=(raw.size,))
        arr[:] = raw[:]; arr.flush()

    man["stages"]["regroup"] = {"molecules": len(keys), "spectra": len(meta),
                                "total_peaks": total}
    save_manifest(cfg, man)
    print(f"[stage B] {len(keys)} molecules, {len(meta)} views, {total} peaks", flush=True)
    return True


# --------------------------------------------- stage C: molecule table, folds, oracle
def _generic_scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ""
    try:
        core = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
    except Exception:
        return ""


def _structure_table(cfg: Config) -> pd.DataFrame:
    """One row per structure, deduplicated per row group so peak memory stays bounded.

    Reading all 2.5 million rows of three string columns at once costs about a
    gigabyte; deduplicating incrementally keeps it to the distinct structures.
    """
    f = pq.ParquetFile(cfg.paths.train_parquet)
    n_rg = f.metadata.num_row_groups
    if cfg.data.limit_row_groups:
        n_rg = min(n_rg, int(cfg.data.limit_row_groups))
    cols = ["inchikey14", "normalized_smiles", "molecular_formula"]
    # Optional: a corpus that ships its own structure-level split carries it here.
    if "provided_fold" in set(f.schema_arrow.names):
        cols.append("provided_fold")
    parts = []
    for g in range(n_rg):
        part = f.read_row_group(g, columns=cols).to_pandas()
        parts.append(part.drop_duplicates("inchikey14"))
    return pd.concat(parts, ignore_index=True).drop_duplicates("inchikey14")


def stage_molecules(cfg: Config, deadline: float) -> bool:
    man = load_manifest(cfg)
    if "molecules" in man["stages"]:
        return True
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog("rdApp.*")

    spectra = pd.read_parquet(cfg.paths.views / "spectra.parquet",
                              columns=["inchikey14", "mol_idx", "instrument_family"])
    raw = _structure_table(cfg)
    mol = (spectra.groupby(["mol_idx", "inchikey14"], sort=True)
           .agg(n_views=("instrument_family", "size"),
                n_timstof=("instrument_family", lambda s: int((s == "timsTOF").sum())))
           .reset_index())
    mol = mol.merge(raw, on="inchikey14", how="left")

    mass, scaf = [], []
    for smi in mol.normalized_smiles:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        mass.append(ExactMolWt(m) if m is not None else np.nan)
        scaf.append(_generic_scaffold(smi) if isinstance(smi, str) else "")
    mol["mono_mass"] = mass
    mol["scaffold"] = scaf

    # A corpus may arrive with folds already assigned. MassSpecGym, for one,
    # splits by MCES edit distance between structures, which is stricter than
    # scaffold grouping and is the split its benchmark is defined on; recomputing
    # our own over it would discard exactly the leakage control we imported it
    # for. When the source supplies a fold per structure we honour it.
    if "provided_fold" in raw.columns and raw.provided_fold.notna().any():
        supplied = raw.set_index("inchikey14").provided_fold
        mol["fold"] = mol.inchikey14.map(supplied).astype("Int64")
        if mol.fold.isna().any():
            missing = int(mol.fold.isna().sum())
            raise ValueError(
                f"{missing} structures have no provided fold; a partially supplied "
                f"split cannot be mixed with a computed one without leaking across "
                f"the boundary")
        mol["fold"] = mol.fold.astype(int)
        return _finish_molecule_table(cfg, mol, man)

    # scaffold-grouped folds, greedy balanced
    key = np.where(mol.scaffold.to_numpy() == "", mol.inchikey14.to_numpy(),
                   mol.scaffold.to_numpy())
    sizes = pd.Series(key).value_counts()
    rng = np.random.RandomState(cfg.data.seed)
    sizes = sizes.sample(frac=1, random_state=rng).sort_values(ascending=False, kind="stable")
    load = np.zeros(cfg.data.n_folds)
    fold_of: Dict[str, int] = {}
    for k, n in sizes.items():
        j = int(load.argmin()); fold_of[k] = j; load[j] += n
    mol["fold"] = [fold_of[k] for k in key]
    return _finish_molecule_table(cfg, mol, man)


def _finish_molecule_table(cfg, mol, man):
    """Oracle flagging, write-out and manifest, shared by both fold routes."""
    if cfg.data.build_oracle:
        oracle = build_oracle_key(cfg)
        mol["is_oracle"] = mol.inchikey14.isin(set(oracle.inchikey14)).astype(bool)
    else:
        mol["is_oracle"] = False

    mol.to_parquet(cfg.paths.molecules)
    man["stages"]["molecules"] = {
        "n": len(mol), "fold_sizes": mol.fold.value_counts().sort_index().tolist(),
        "oracle_structures": int(mol.is_oracle.sum())}
    save_manifest(cfg, man)
    print(f"[stage C] {len(mol)} molecules; oracle structures {int(mol.is_oracle.sum())}",
          flush=True)
    return True


def build_oracle_key(cfg: Config) -> pd.DataFrame:
    """Recover the test answer key by exact duplicate matching (see module docstring)."""
    out = cfg.paths.oracle
    if out.exists():
        return pd.read_csv(out)

    def h(mz, it) -> str:
        return hashlib.md5(np.asarray(mz, "float64").tobytes()
                           + np.asarray(it, "float64").tobytes()).hexdigest()

    test = pq.read_table(cfg.paths.test_parquet,
                         columns=["molecule_id", "spectrum_id", "precursor_mz",
                                  "ms2_mzs", "ms2_normalized_intensities"]).to_pandas()
    test["h"] = [h(m, i) for m, i in zip(test.ms2_mzs, test.ms2_normalized_intensities)]
    want = set(test.h)
    pset = list(set(np.round(test.precursor_mz.to_numpy(), 4)))

    f = pq.ParquetFile(cfg.paths.train_parquet)
    n_rg = f.metadata.num_row_groups
    if cfg.data.limit_row_groups:
        n_rg = min(n_rg, int(cfg.data.limit_row_groups))
    rows: List[tuple] = []
    for g in range(n_rg):
        tb = f.read_row_group(g, columns=["inchikey14", "normalized_smiles",
                                          "molecular_formula", "ingest_lib",
                                          "precursor_mz", "ms2_mzs",
                                          "ms2_normalized_intensities"])
        p = np.round(tb.column("precursor_mz").to_numpy(), 4)
        cand = np.where(np.isin(p, pset))[0]
        if cand.size == 0:
            continue
        mzs, its = tb.column("ms2_mzs"), tb.column("ms2_normalized_intensities")
        for i in cand:
            i = int(i)
            hh = h(mzs[i].as_py(), its[i].as_py())
            if hh in want:
                rows.append((hh, tb.column("inchikey14")[i].as_py(),
                             tb.column("normalized_smiles")[i].as_py(),
                             tb.column("molecular_formula")[i].as_py(),
                             tb.column("ingest_lib")[i].as_py()))
        if len({r[0] for r in rows}) == len(want):
            break

    m = pd.DataFrame(rows, columns=["h", "inchikey14", "smiles", "formula", "lib"]
                     ).drop_duplicates("h")
    j = test[["molecule_id", "spectrum_id", "h"]].merge(m, on="h", how="left")
    key = (j.dropna(subset=["inchikey14"])
             .groupby("molecule_id")
             .agg(inchikey14=("inchikey14", "first"), smiles=("smiles", "first"),
                  formula=("formula", "first"),
                  n_matched=("spectrum_id", "size"),
                  n_conflict=("inchikey14", "nunique"))
             .reset_index())
    out.parent.mkdir(parents=True, exist_ok=True)
    key.to_csv(out, index=False)
    print(f"[oracle] resolved {len(key)}/{test.molecule_id.nunique()} test molecules; "
          f"conflicts {(key.n_conflict > 1).sum()}", flush=True)
    return key


# ------------------------------------------------------------- stage D: the targets
def stage_targets(cfg: Config, deadline: float) -> bool:
    man = load_manifest(cfg)
    if "targets" in man["stages"]:
        return True
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    from .brics import brics_pieces

    mol = pd.read_parquet(cfg.paths.molecules)
    fit_mask = mol.fold.isin(cfg.train_folds_for_vocab()).to_numpy()

    # pass 1: BRICS fragments, vocabulary fitted on training folds only
    frags: List[List[str]] = []
    counter: Dict[str, int] = {}
    for smi, in_fit in zip(mol.normalized_smiles, fit_mask):
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        pieces, _ = brics_pieces(m)
        frags.append(pieces)
        if in_fit:
            for p in set(pieces):
                counter[p] = counter.get(p, 0) + 1

    vocab = [p for p, c in sorted(counter.items(), key=lambda kv: -kv[1])
             if c >= cfg.data.min_fit_molecules][: cfg.data.brics_top_k]
    vidx = {p: i for i, p in enumerate(vocab)}

    D = np.zeros((len(mol), len(vocab)), dtype=np.int8)
    for i, pieces in enumerate(frags):
        for p in pieces:
            j = vidx.get(p)
            if j is not None:
                D[i, j] = 1

    # pass 2: functional-group counts
    fr_names = [n for n, _ in Descriptors.descList if n.startswith("fr_")]
    fr_funcs = [f for n, f in Descriptors.descList if n.startswith("fr_")]
    N = np.zeros((len(mol), len(fr_names) + 3), dtype=np.int16)
    for i, (smi, pieces) in enumerate(zip(mol.normalized_smiles, frags)):
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if m is not None:
            for j, fn in enumerate(fr_funcs):
                try:
                    N[i, j] = int(fn(m))
                except Exception:
                    N[i, j] = 0
        N[i, len(fr_names)] = len(pieces)
        N[i, len(fr_names) + 1] = len(set(pieces))
        N[i, len(fr_names) + 2] = max(len(pieces) - 1, 0)

    prev = D[fit_mask].mean(0) if fit_mask.any() else D.mean(0)
    pres_cols = [f"brics_{i:04d}" for i in range(len(vocab))]
    cnt_cols = fr_names + ["n_pieces", "n_distinct_pieces", "n_cuts"]
    out = pd.DataFrame(D, columns=pres_cols)
    out[cnt_cols] = N
    out.insert(0, "inchikey14", mol.inchikey14.to_numpy())
    out.insert(1, "mol_idx", mol.mol_idx.to_numpy())
    out.to_parquet(cfg.paths.targets)

    pd.DataFrame({
        "variable": pres_cols + cnt_cols,
        "family": ["brics_presence"] * len(pres_cols) + ["count"] * len(cnt_cols),
        "pattern": vocab + cnt_cols,
        "fit_prevalence": list(prev) + [float((N[fit_mask, j] > 0).mean())
                                        for j in range(N.shape[1])],
        "keep_for_bayes": list(prev >= cfg.data.min_prevalence)
                          + [True] * len(cnt_cols),
    }).to_csv(cfg.paths.target_dict, index=False)

    man["stages"]["targets"] = {"n_presence": len(pres_cols), "n_count": len(cnt_cols),
                                "above_min_prevalence": int((prev >= cfg.data.min_prevalence).sum())}
    save_manifest(cfg, man)
    print(f"[stage D] {len(pres_cols)} presence + {len(cnt_cols)} count targets; "
          f"{int((prev >= cfg.data.min_prevalence).sum())} above prevalence "
          f"{cfg.data.min_prevalence}", flush=True)
    return True


# ---------------------------------------------------------------------------- main
def _check_scope(cfg: Config) -> None:
    """Refuse to extend a preparation that was made with a different row-group scope.

    A smoke run with --max-row-groups writes a manifest whose stages read as
    complete. Continuing into it would silently train on a fraction of the data,
    which is the kind of error that never announces itself.
    """
    man = load_manifest(cfg)
    prior = man.get("limit_row_groups", "unset")
    current = cfg.data.limit_row_groups
    if prior == "unset":
        man["limit_row_groups"] = current
        save_manifest(cfg, man)
        return
    if prior != current:
        raise SystemExit(
            f"{cfg.paths.prepared} was prepared with limit_row_groups={prior!r} but "
            f"this run asks for {current!r}. Delete or rename that directory and "
            f"start again; mixing the two would train on a subset without saying so.")


def run(cfg: Config, budget: float = float("inf")) -> bool:
    cfg.paths.prepared.mkdir(parents=True, exist_ok=True)
    _check_scope(cfg)
    deadline = time.time() + budget
    for fn in (stage_spectra, stage_regroup, stage_molecules, stage_targets):
        if not fn(cfg, deadline):
            return False
    print("STAGE 0 COMPLETE", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--budget", type=float, default=float("inf"),
                    help="seconds; rerun the same command to resume")
    ap.add_argument("--max-peaks", type=int, default=None,
                    help="0 disables the cap (specification default)")
    a = ap.parse_args()
    cfg = Config()
    cfg.paths.root = a.root
    if a.max_peaks is not None:
        cfg.data.max_peaks = a.max_peaks or 0
    run(cfg, a.budget)


if __name__ == "__main__":
    main()
