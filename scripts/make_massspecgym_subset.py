"""Build a dbf2 corpus from MassSpecGym.

MassSpecGym (Bushuiev et al., NeurIPS 2024 Datasets and Benchmarks) is MIT
licensed and downloadable without an account, which the CASMI 2026 aggregate is
not. It is also split by MCES edit distance rather than by scaffold, so
structurally near-identical molecules cannot straddle the train/test boundary.
Those two properties are the reason this corpus exists here: it is the one
dataset in this project that a third party can obtain and reproduce exactly.

    python scripts/make_massspecgym_subset.py --root .
    python -m dbf2 prepare --root . --dataset msg
    python -m dbf2 ablate  --root . --dataset msg --rungs M0 M1D M1R M1 M2

What this script does NOT do is re-derive the split. MassSpecGym's folds are
carried through as ``provided_fold`` and honoured by ``prepare``; recomputing our
own scaffold folds over them would discard the leakage control we came for. Its
three folds map onto this project's convention as

    test  -> fold 0   (held out, all reported numbers)
    val   -> fold 1   (checkpoint selection)
    train -> folds 2, 3, 4 (fitting; used together, split only for symmetry)

Caveats worth knowing before comparing against the other corpora. MassSpecGym is
far more homogeneous: around 98% of its spectra are Orbitrap or QTOF and its
adducts are dominated by [M+H]+ and [M+Na]+, against 66 instrument types and 112
adducts in the open CASMI subset. Collision energy is recorded as NCE and is
missing for a substantial minority of rows, which the covariate encoder handles
through its explicit missing indicator. Both facts matter for any claim about
acquisition metadata, because there is less acquisition variation here to explain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HF = ("https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/"
      "resolve/main/data/{name}")
VERSIONS = {"1.0": "MassSpecGym.tsv", "1.5": "MassSpecGym1.5.tsv"}

# The schema dbf2.prepare reads. Everything below exists to produce exactly this.
OUT_COLS = ["inchikey14", "normalized_smiles", "molecular_formula", "ingest_lib",
            "ionization_mode", "instrument_type", "adduct", "precursor_mz",
            "precursor_error_ppm", "ms2_mzs", "ms2_normalized_intensities",
            "collision_energy_ev", "collision_energy_orig_units", "provided_fold"]

FOLD_MAP = {"test": 0, "val": 1}          # train is spread over 2, 3, 4 below


def polarity_of(adduct: str) -> str:
    """MassSpecGym gives no polarity column; the adduct's charge carries it."""
    if not isinstance(adduct, str) or not adduct:
        return "unknown"
    tail = adduct.rstrip()
    if tail.endswith("-"):
        return "negative"
    if tail.endswith("+"):
        return "positive"
    return "unknown"


def parse_peaks(s):
    if not isinstance(s, str) or not s:
        return np.empty(0, np.float64)
    return np.fromstring(s, sep=",", dtype=np.float64)


def download(url: str, dest: Path) -> None:
    """Stream to disk. Kept dependency-free on purpose: this script is the one a
    reader runs before they have anything else installed."""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB", end="")
    print()
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--version", default="1.0", choices=sorted(VERSIONS),
                    help="1.0 is the version the NeurIPS 2024 paper describes")
    ap.add_argument("--keep-tsv", action="store_true",
                    help="keep the downloaded TSV instead of only the parquet")
    a = ap.parse_args()

    root = Path(a.root)
    name = VERSIONS[a.version]
    tsv = root / "data" / name
    out = root / "data" / "train_msg.parquet"

    if not tsv.exists():
        download(HF.format(name=name), tsv)
    else:
        print(f"  using cached {tsv}")

    sha = hashlib.sha256(tsv.read_bytes()).hexdigest()
    print(f"  sha256 {sha}")

    df = pd.read_csv(tsv, sep="\t")
    print(f"  {len(df):,} spectra, {df.inchikey.nunique():,} structures")

    bad_fold = set(df.fold.unique()) - {"train", "val", "test"}
    if bad_fold:
        raise ValueError(f"unexpected fold labels {sorted(bad_fold)}")

    # train molecules spread over folds 2-4. They are always used together, so the
    # split is cosmetic; it is done per structure so no structure lands in two.
    keys = np.sort(df.loc[df.fold == "train", "inchikey"].unique())
    train_fold = {k: 2 + (i % 3) for i, k in enumerate(keys)}
    fold = [FOLD_MAP.get(f, train_fold.get(k, -1))
            for f, k in zip(df.fold, df.inchikey)]
    if -1 in fold:
        raise ValueError("a spectrum was left without a fold")

    o = pd.DataFrame({
        "inchikey14": df.inchikey.astype(str),
        "normalized_smiles": df.smiles.astype(str),
        "molecular_formula": df.formula.astype(str),
        "ingest_lib": f"massspecgym-{a.version}",
        "ionization_mode": [polarity_of(x) for x in df.adduct],
        "instrument_type": df.instrument_type.astype(str),
        "adduct": df.adduct.astype(str),
        "precursor_mz": df.precursor_mz.astype(np.float64),
        # MassSpecGym reports no per-spectrum calibration error. NaN is kept by the
        # precursor filter, so this neither drops rows nor fakes a tolerance.
        "precursor_error_ppm": np.full(len(df), np.nan, np.float64),
        "ms2_mzs": [parse_peaks(s) for s in df.mzs],
        "ms2_normalized_intensities": [parse_peaks(s) for s in df.intensities],
        "collision_energy_ev": df.collision_energy.astype(np.float64),
        # Normalised collision energy; harmonise_energy converts it with the
        # singly charged Thermo convention eV = NCE * m/z / 500. Mislabelling this
        # as eV would put two different scales on one axis.
        "collision_energy_orig_units": "NCE",
        "provided_fold": np.asarray(fold, np.int32),
    })[OUT_COLS]

    out.parent.mkdir(parents=True, exist_ok=True)
    o.to_parquet(out, index=False)

    prov = {
        "source": "MassSpecGym",
        "version": a.version,
        "file": name,
        "url": HF.format(name=name),
        "sha256": sha,
        "licence": "MIT",
        "retrieved_utc": pd.Timestamp.utcnow().isoformat(),
        "citation": ("Bushuiev R et al. (2024) MassSpecGym: a benchmark for the "
                     "discovery and identification of molecules. NeurIPS 2024, "
                     "Datasets and Benchmarks Track."),
        "n_spectra": int(len(o)),
        "n_structures": int(o.inchikey14.nunique()),
        "fold_sizes": {str(k): int(v) for k, v in
                       o.provided_fold.value_counts().sort_index().items()},
        "split": "MCES edit distance, threshold 10, as published with the benchmark",
        "collision_energy_missing": float(o.collision_energy_ev.isna().mean()),
        "instrument_types": int(o.instrument_type.nunique()),
        "adducts": int(o.adduct.nunique()),
    }
    (root / "data" / "train_msg_provenance.json").write_text(
        json.dumps(prov, indent=2), encoding="utf-8")

    if not a.keep_tsv:
        tsv.unlink()

    print(f"  wrote {out}")
    print(f"  folds {prov['fold_sizes']}  "
          f"instruments {prov['instrument_types']}  adducts {prov['adducts']}  "
          f"CE missing {prov['collision_energy_missing']:.1%}")


if __name__ == "__main__":
    main()
