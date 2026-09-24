"""Uncertainty for the contrasts against the TUNED dense baseline.

Sect. 5b reports that every peak-token arm sits below the tuned baseline, with a
seed standard deviation and nothing else. Every other contrast in the paper
carries two intervals; this one carried none, and it is the claim in the title.

Nothing here requires retraining. The tuned baseline's fold-0 predictions are
saved for all three seeds, so both intervals the paper already defines can be
computed from what is on disk:

  (ii) target-resampling  - 5,000 draws over the scored targets, after averaging
       each target's average precision across the three seeds, as in
       scripts/seed_sweep_analysis.py;
  (iii) scaffold-cluster  - 2,000 draws over the scaffold clusters partitioning
       the held-out structures, macro AUPRC recomputed per arm inside each draw
       and the contrast formed inside the draw, as in
       scripts/scaffold_cluster_bootstrap.py.

Both estimators are copied from those scripts unchanged, including the rule that
a target enters the macro average only while it retains at least `min_positives`
positives in the resample. The centring check is retained: if the bootstrap mean
does not sit near the point estimate the resample is not estimating the reported
quantity and the interval must not be reported.

    python scripts/tuned_baseline_intervals.py --root . --dataset open
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = {"2026": "", "7": "_s7", "13": "_s13"}

# M0T is the tuned dense baseline (lr 3e-3, width 1024, dropout 0.3); M0 is the
# metadata-matched but untuned one every other table in the paper uses.
CONTRASTS = [("M0T-M0",  "M0T", "M0",  "comparator tuning"),
             ("M1D-M0T", "M1D", "M0T", "composite"),
             ("M1R-M0T", "M1R", "M0T", "composite"),
             ("M1-M0T",  "M1",  "M0T", "composite"),
             ("M2-M0T",  "M2",  "M0T", "composite")]


def ap_matrix(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Average precision per column, vectorised, matching dbf2.evaluate."""
    order = np.argsort(-p, axis=0, kind="stable")
    ys = np.take_along_axis(y, order, axis=0)
    tp = np.cumsum(ys, axis=0)
    prec = tp / np.arange(1, y.shape[0] + 1, dtype=np.float64)[:, None]
    n_pos = ys.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (prec * ys).sum(axis=0) / n_pos


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--root", default=".")
    ap_.add_argument("--dataset", default="open", choices=["default", "open", "msg"])
    ap_.add_argument("--draws", type=int, default=2000)
    ap_.add_argument("--target-draws", type=int, default=5000)
    ap_.add_argument("--seed", type=int, default=2026)
    ap_.add_argument("--min-positives", type=int, default=5)
    a = ap_.parse_args()

    root = Path(a.root)
    prep = {"default": "dbf2_prepared", "open": "dbf2_prepared_open",
            "msg": "dbf2_prepared_msg"}[a.dataset]
    runs = {"default": "dbf2_runs", "open": "dbf2_runs_open",
            "msg": "dbf2_runs_msg"}[a.dataset]
    arms = sorted({x for _, hi, lo, _ in CONTRASTS for x in (hi, lo)})

    # target and the scored mask first, so each arm's probabilities can be sliced
    # to the scored targets as they are loaded rather than held in full
    d0 = np.load(root / runs / "M1" / "M1_test_predictions.npz", allow_pickle=True)
    target = d0["target"].astype(np.float64)
    groups = d0["group_id"]
    pos = target.sum(axis=0)
    scored = (pos >= a.min_positives) & (pos <= target.shape[0] - a.min_positives)
    target = target[:, scored]
    print(f"  {target.shape[0]:,} held-out molecules, {int(scored.sum())} scored targets")

    prob = {}
    for arm in arms:
        stack = []
        for suf in SEEDS.values():
            f = root / runs / f"{arm}{suf}" / f"{arm}{suf}_test_predictions.npz"
            stack.append(np.load(f, allow_pickle=True)["prob"].astype(np.float64)[:, scored])
        prob[arm] = np.stack(stack)            # (n_seeds, n_molecules, n_scored)
        print(f"    loaded {arm} ({prob[arm].shape[0]} seeds)")

    # per-target AP averaged over seeds, and the full-set macro per arm
    apm = {arm: np.mean([ap_matrix(target, prob[arm][si])
                         for si in range(prob[arm].shape[0])], axis=0)
           for arm in arms}
    macro_full = {arm: float(np.nanmean(apm[arm])) for arm in arms}
    print("\n  point estimates (macro AUPRC, 3-seed mean of per-target AP):")
    for arm in arms:
        print(f"    {arm:4s} {macro_full[arm]:.4f}")

    out = {"dataset": a.dataset, "n_molecules": int(target.shape[0]),
           "n_scored_targets": int(scored.sum()),
           "target_draws": a.target_draws, "scaffold_draws": a.draws,
           "macro_auprc": macro_full, "contrasts": {}}

    # ------------------------------------------------ (ii) target resampling
    rng = np.random.default_rng(0)
    print("\n  target-resampling intervals "
          f"({a.target_draws:,} draws over {int(scored.sum())} scored targets):")
    for name, hi, lo, kind in CONTRASTS:
        d = apm[hi] - apm[lo]
        idx = rng.integers(0, d.size, (a.target_draws, d.size))
        dr = np.nanmean(d[idx], axis=1)
        rec = {"kind": kind, "point": float(np.nanmean(d)),
               "target_lo": float(np.quantile(dr, 0.025)),
               "target_hi": float(np.quantile(dr, 0.975)),
               "frac_improved": float(np.nanmean(d > 0))}
        out["contrasts"][name] = rec
        print(f"    {name:9s} {rec['point']:+.4f}  "
              f"[{rec['target_lo']:+.4f}, {rec['target_hi']:+.4f}]  "
              f"{rec['frac_improved']:.1%} of targets")

    # ------------------------------------------------ per-seed, for the ordering check
    print("\n  per seed:")
    for name, hi, lo, _ in CONTRASTS:
        vals = []
        for si in range(prob[hi].shape[0]):
            mh = float(np.nanmean(ap_matrix(target, prob[hi][si])))
            ml = float(np.nanmean(ap_matrix(target, prob[lo][si])))
            vals.append(mh - ml)
        out["contrasts"][name]["per_seed"] = vals
        out["contrasts"][name]["seed_sd"] = float(np.std(vals, ddof=1))
        agree = sum(v > 0 for v in vals)
        print(f"    {name:9s} {', '.join(f'{v:+.4f}' for v in vals)}   "
              f"{agree}/{len(vals)} positive")

    # ------------------------------------------------ (iii) scaffold clusters
    mol = pd.read_parquet(root / prep / "molecules.parquet").set_index("mol_idx")
    scaf = mol.scaffold.reindex(groups).to_numpy()
    ikey = mol.inchikey14.reindex(groups).to_numpy()
    key = np.where(pd.isna(scaf) | (scaf == ""), ikey, scaf)
    uniq, cluster = np.unique(key, return_inverse=True)
    members = [np.flatnonzero(cluster == c) for c in range(uniq.size)]
    print(f"\n  {uniq.size:,} scaffold clusters; {a.draws:,} draws")

    rng = np.random.default_rng(a.seed)
    draws = {k: np.empty(a.draws) for k, _, _, _ in CONTRASTS}
    for b in range(a.draws):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([members[c] for c in pick])
        y = target[idx]
        n_pos = y.sum(axis=0)
        keep = (n_pos >= a.min_positives) & (n_pos <= y.shape[0] - a.min_positives)
        yk = y[:, keep]
        macro = {arm: float(np.nanmean(
                    np.mean([ap_matrix(yk, prob[arm][si][idx][:, keep])
                             for si in range(prob[arm].shape[0])], axis=0)))
                 for arm in arms}
        for name, hi, lo, _ in CONTRASTS:
            draws[name][b] = macro[hi] - macro[lo]
        if (b + 1) % 100 == 0:
            print(f"\r  {b + 1}/{a.draws} draws", end="", flush=True)
    print()

    print("\n  centring check (bootstrap mean vs point estimate):")
    worst = 0.0
    for name, hi, lo, _ in CONTRASTS:
        point = macro_full[hi] - macro_full[lo]
        gap = abs(draws[name].mean() - point)
        worst = max(worst, gap)
        flag = "" if gap < 0.25 * max(abs(point), 1e-6) else "   <-- OFF-CENTRE"
        print(f"    {name:9s} point {point:+.4f}  boot {draws[name].mean():+.4f}"
              f"  gap {gap:+.4f}{flag}")

    print("\n  scaffold-cluster intervals:")
    for name, _, _, _ in CONTRASTS:
        d = draws[name]
        lo_, hi_ = np.percentile(d, [2.5, 97.5])
        out["contrasts"][name].update({"scaffold_mean": float(d.mean()),
                                       "scaffold_sd": float(d.std(ddof=1)),
                                       "scaffold_lo": float(lo_),
                                       "scaffold_hi": float(hi_)})
        print(f"    {name:9s} {d.mean():+.4f}  [{lo_:+.4f}, {hi_:+.4f}]")
    out["centring_worst_gap"] = float(worst)
    if worst > 0.01:
        print("\n  WARNING: does not centre on the point estimates; do not report.")

    dest = root / "results" / "tuned_baseline_intervals.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {dest}")


if __name__ == "__main__":
    main()
