"""Candidate ranking and submission (specification sections 7, 9 and 12.1).

    S_lambda(G; X) = lambda_B s_B(X,G) + lambda_M s_M(X,G) + M_valid(G)

with nonnegative lambda fitted on held-out validation queries, never set by hand.

    python -m dbf2 predict --root . --run dbf2_runs/M2 --budget 150

Resumable: the formula enumeration is cached per query, so rerunning continues.

The recovered test key is NEVER read on the prediction path. It is used only by the
optional evaluation at the end, which is reported separately and must not be
confused with a competition result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Config, InferenceConfig
from .candidates import score_formulas, candidate_recall
from .dataset import MoleculeViewDataset, ViewStore, collate, load_targets, split_molecules
from .model import DeepBayesFrag
from .retrieval import GeneralizedPosterior, evaluate_retrieval
from . import chem


# --------------------------------------------------------------- candidate library
class CandidateLibrary:
    """Structures plus their packed BRICS presence vectors, indexed by mass and formula."""

    def __init__(self, root: Path, n_targets: int):
        d = root / "candidates"
        self.lib = pd.read_parquet(d / "library.parquet")
        self.packed = np.load(d / "presence_packed.npy")
        self.n_targets = n_targets
        self.key = self.lib.inchikey14.to_numpy()
        self.smiles = self.lib.normalized_smiles.to_numpy()
        self.formula = self.lib.molecular_formula.to_numpy()
        self.mass = self.lib.mono_mass.to_numpy(np.float64)
        self._order = np.argsort(self.mass)
        self._sorted = self.mass[self._order]
        idx: Dict[str, List[int]] = {}
        for i, f in enumerate(self.formula):
            idx.setdefault(f, []).append(i)
        self._formula = {k: np.asarray(v, np.int64) for k, v in idx.items()}

    def descriptors(self, rows: np.ndarray) -> np.ndarray:
        return np.unpackbits(self.packed[rows], axis=1)[:, : self.n_targets].astype(np.float64)

    def by_mass(self, m: float, ppm: float) -> np.ndarray:
        tol = m * ppm * 1e-6
        lo, hi = np.searchsorted(self._sorted, [m - tol, m + tol])
        return self._order[lo:hi]

    def by_formula(self, f: str) -> np.ndarray:
        return self._formula.get(f, np.empty(0, np.int64))


# ------------------------------------------------------------------- model outputs
@torch.no_grad()
def presence_probabilities(model, store: ViewStore, ids, cfg: Config, device
                           ) -> Tuple[np.ndarray, np.ndarray]:
    ds = MoleculeViewDataset(store, ids, max_peaks=cfg.data.max_peaks or 0,
                             max_views=cfg.train.views_per_molecule, train=False, seed=0)
    dl = DataLoader(ds, batch_size=cfg.train.batch_molecules, shuffle=False,
                    collate_fn=collate)
    P, G = [], []
    for b in dl:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        P.append(torch.sigmoid(model(b)["logit"].float()).cpu())
        G.append(b["group_id"].cpu())
    return torch.cat(P).numpy(), torch.cat(G).numpy()


# ------------------------------------------------------------------- query assembly
def formula_cache(cfg: Config, tag: str) -> Path:
    p = cfg.paths.prepared / "formula_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{tag}.parquet"


def build_formula_scores(queries: pd.DataFrame, peaks: Dict, icfg: InferenceConfig,
                         cache: Path, deadline: float, top_formulas: int = 5
                         ) -> Optional[pd.DataFrame]:
    """Enumerate and score formulas per query, resumably."""
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(
        columns=["query", "formula", "prob"])
    have = set(done["query"].unique())
    rows = [done] if len(done) else []
    todo = [q for q in queries["query"] if q not in have]
    for n, q in enumerate(todo):
        if time.time() > deadline:
            if rows:
                pd.concat(rows, ignore_index=True).to_parquet(cache)
            print(f"[predict] formula stage: {len(have)}/{len(queries)} done; rerun to resume",
                  flush=True)
            return None
        r = queries[queries["query"] == q].iloc[0]
        mz, it = peaks[q]
        fs = score_formulas(float(r.precursor_mz), r.adduct, mz, it, icfg)
        if len(fs):
            fs = fs.head(top_formulas)[["formula", "prob"]].copy()
            fs.insert(0, "query", q)
            rows.append(fs)
        have.add(q)
        if (n + 1) % 50 == 0:
            pd.concat(rows, ignore_index=True).to_parquet(cache)
            print(f"[predict] formula stage {len(have)}/{len(queries)}", flush=True)
    out = pd.concat(rows, ignore_index=True) if rows else done
    out.to_parquet(cache)
    return out


def assemble(lib: CandidateLibrary, r, fs: pd.DataFrame, probs: np.ndarray,
             icfg: InferenceConfig) -> Dict[str, np.ndarray]:
    """Candidate rows and their evidence terms for one query."""
    add = chem.get_adduct(r.adduct)
    if add is None:
        return {}
    rows, logpm = [], []
    for f, p in zip(fs.formula, fs.prob):
        idx = lib.by_formula(f)
        if idx.size:
            rows.append(idx)
            logpm.append(np.full(idx.size, np.log(max(float(p), 1e-12))))
    neutral = add.neutral_mass(float(r.precursor_mz))
    extra = lib.by_mass(neutral, icfg.candidate_ppm)
    if rows:
        cur = np.concatenate(rows)
        extra = np.setdiff1d(extra, cur)
        rows.append(extra); logpm.append(np.full(extra.size, np.log(1e-4)))
    else:
        rows, logpm = [extra], [np.full(extra.size, np.log(1e-4))]
    idx = np.concatenate(rows); lp = np.concatenate(logpm)
    uniq, first = np.unique(idx, return_index=True)
    keep = np.sort(first)
    idx, lp = idx[keep], lp[keep]
    if idx.size > icfg.max_candidates:
        sel = np.argsort(-lp)[: icfg.max_candidates]
        idx, lp = idx[sel], lp[sel]
    if idx.size == 0:
        return {}
    d = lib.descriptors(idx)
    p = np.clip(probs, 1e-3, 1 - 1e-3)
    s_b = d @ np.log(p) + (1.0 - d) @ np.log1p(-p)
    return {"rows": idx, "bayes": s_b, "formula": lp}


# ------------------------------------------------------------------------ the run
def run(cfg: Config, run_dir: Path, budget: float, n_calib: int) -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deadline = time.time() + budget
    icfg = cfg.infer

    presence, counts, row_of, pres_names, cnt_names = load_targets(cfg)
    cfg.model.n_presence, cfg.model.n_count = presence.shape[1], counts.shape[1]
    ck = sorted(run_dir.glob("*_best.pt"))
    if not ck:
        raise FileNotFoundError(f"no checkpoint in {run_dir}")
    model = DeepBayesFrag(cfg.model).to(device)
    model.load_state_dict(torch.load(ck[0], map_location=device, weights_only=False)["model"])
    model.eval()

    lib = CandidateLibrary(cfg.paths.prepared, presence.shape[1])
    print(f"[predict] candidate library: {len(lib.key)} structures", flush=True)

    # ---- validation queries, used only to fit lambda
    mol = pd.read_parquet(cfg.paths.molecules)
    val = mol[(mol.fold == cfg.train.val_fold) & (~mol.is_oracle)]
    rng = np.random.RandomState(cfg.train.seed)
    val = val.iloc[rng.choice(len(val), min(n_calib, len(val)), replace=False)]
    vstore = ViewStore(cfg.paths.views, "train")
    vprob, vids = presence_probabilities(model, vstore, val.mol_idx.to_numpy(), cfg, device)
    vmeta = vstore.meta.groupby("mol_idx").first()
    vq = pd.DataFrame({"query": vids,
                       "precursor_mz": vmeta.loc[vids, "precursor_mz"].to_numpy(),
                       "adduct": vmeta.loc[vids, "adduct"].to_numpy()})
    vpeaks = {}
    for g in vids:
        s, e = vstore.span[int(g)]
        best = s + int(np.argmax(vstore.n_peaks[s:e]))
        vpeaks[g] = vstore.peaks(best)
    vfs = build_formula_scores(vq, vpeaks, icfg, formula_cache(cfg, "val"), deadline)
    if vfs is None:
        return False

    truth_key = dict(zip(mol.mol_idx, mol.inchikey14))
    gp = GeneralizedPosterior({"bayes": 1.0, "formula": 1.0})
    terms, truth_pos = [], []
    for i, g in enumerate(vids):
        r = vq.iloc[i]
        f = vfs[vfs["query"] == g]
        a = assemble(lib, r, f, vprob[i], icfg)
        if not a:
            continue
        pos = np.where(lib.key[a["rows"]] == truth_key[int(g)])[0]
        terms.append({"bayes": a["bayes"], "formula": a["formula"]})
        truth_pos.append(int(pos[0]) if pos.size else -1)
    w = gp.calibrate(terms, truth_pos, n_steps=300, lr=0.05)
    hit = sum(1 for t in truth_pos if t >= 0)
    print(f"[predict] lambda fitted on {len(terms)} validation queries "
          f"({hit} with the truth in range): {({k: round(v, 3) for k, v in w.items()})}",
          flush=True)

    # ---- validation retrieval, an honest estimate before the test set is touched
    vrank, vtruth, vpool = [], [], []
    for i, g in enumerate(vids):
        r = vq.iloc[i]
        a = assemble(lib, r, vfs[vfs["query"] == g], vprob[i], icfg)
        if not a:
            continue
        s = gp.score({"bayes": a["bayes"], "formula": a["formula"]})
        vrank.append(list(lib.key[a["rows"]][np.argsort(-s)[: icfg.top_k]]))
        vpool.append(list(lib.key[a["rows"]]))       # the FULL pool, not the top k
        vtruth.append(truth_key[int(g)])
    vres = evaluate_retrieval(vrank, vtruth)
    # recall is measured on the full candidate pool: it is the ceiling on MRR@25,
    # and measuring it on the truncated list would merely restate Top-25
    vres["candidate"] = candidate_recall(vpool, vtruth)
    print(f"[predict] validation retrieval: {json.dumps(vres, indent=1)}", flush=True)

    # ---- test predictions
    tstore = ViewStore(cfg.paths.views, "test")
    tprob, tids = presence_probabilities(model, tstore, tstore.groups, cfg, device)
    names = np.array(tstore.id_of_index)
    tmeta = tstore.meta.groupby(tstore.group_key).first()
    tq = pd.DataFrame({"query": tids,
                       "precursor_mz": tmeta.loc[tids, "precursor_mz"].to_numpy(),
                       "adduct": tmeta.loc[tids, "adduct"].to_numpy()})
    tpeaks = {}
    for g in tids:
        s, e = tstore.span[int(g)]
        best = s + int(np.argmax(tstore.n_peaks[s:e]))
        tpeaks[g] = tstore.peaks(best)
    tfs = build_formula_scores(tq, tpeaks, icfg, formula_cache(cfg, "test"), deadline)
    if tfs is None:
        return False

    sub, diag = [], []
    for i, g in enumerate(tids):
        r = tq.iloc[i]
        a = assemble(lib, r, tfs[tfs["query"] == g], tprob[i], icfg)
        if not a:
            sub.append((names[int(g)], ";".join(["C"] * icfg.top_k)))
            diag.append((names[int(g)], 0, 0)); continue
        s = gp.score({"bayes": a["bayes"], "formula": a["formula"]})
        top = a["rows"][np.argsort(-s)[: icfg.top_k]]
        smi = list(lib.smiles[top])
        while len(smi) < icfg.top_k:
            smi.append(smi[-1] if smi else "C")
        sub.append((names[int(g)], ";".join(smi)))
        diag.append((names[int(g)], len(a["rows"]), len(top)))

    out = run_dir / "submission.csv"
    pd.DataFrame(sub, columns=["molecule_id", "smiles"]).to_csv(out, index=False)
    pd.DataFrame(diag, columns=["molecule_id", "n_candidates", "n_ranked"]).to_csv(
        run_dir / "test_diagnostics.csv", index=False)
    json.dump({"lambda": w, "validation": vres,
               "n_test": len(sub), "library": int(len(lib.key))},
              open(run_dir / "predict_report.json", "w"), indent=2)
    print(f"[predict] wrote {out}", flush=True)
    print("PREDICT COMPLETE", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--budget", type=float, default=float("inf"))
    ap.add_argument("--calibration-queries", type=int, default=400)
    a = ap.parse_args()
    cfg = Config.ablation("M2", a.root)
    run(cfg, a.run, a.budget, a.calibration_queries)


if __name__ == "__main__":
    main()
