"""Scaffold-cluster bootstrap over the held-out molecules.

The target-resampling intervals reported elsewhere in this work resample BRICS
targets after averaging average precision across seeds. They therefore describe
heterogeneity across the scored target set and say nothing about the held-out
molecules: the same 10,058 structures enter every resample. This script supplies
the complementary interval, resampling the held-out set itself.

Scaffold groups, not molecules, are the resampling unit. Molecules sharing a
generic Murcko scaffold were assigned to a fold together, so they are not
exchangeable with one another; resampling individual molecules would treat
dependent observations as independent and understate the interval. Drawing whole
scaffold groups with replacement preserves that dependence.

Within each resample the full macro AUPRC is recomputed per arm over the scored
targets, and the contrasts are formed inside the resample, so the paired
structure of the comparison is retained.

    python scripts/scaffold_cluster_bootstrap.py --root . --dataset open --draws 2000

Neither this interval nor the target-resampling one is a confidence interval for
the population of all MS/MS tasks. This one covers the sampling of held-out
structures at a fixed target set; the other covers the sampling of targets at a
fixed held-out set; seed spread is reported separately again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = {"2026": "", "7": "_s7", "13": "_s13"}
CONTRASTS = [("M1D-M0", "M1D", "M0", "composite"),
             ("M1R-M1D", "M1R", "M1D", "composite"),
             ("M1R-M0", "M1R", "M0", "composite"),
             ("M1-M1R", "M1", "M1R", "one-factor"),
             ("M2-M1", "M2", "M1", "one-factor"),
             ("M1-M0", "M1", "M0", "composite"),
             ("M2-M0", "M2", "M0", "composite")]


def ap_matrix(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Average precision per column, vectorised, matching dbf2.evaluate.

    Same step-function estimator: sort each column by descending score, take the
    running precision at every positive, divide by the positive count.
    """
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
    ap_.add_argument("--seed", type=int, default=2026)
    ap_.add_argument("--min-positives", type=int, default=5)
    a = ap_.parse_args()

    root = Path(a.root)
    prep = {"default": "dbf2_prepared", "open": "dbf2_prepared_open",
            "msg": "dbf2_prepared_msg"}[a.dataset]
    runs = {"default": "dbf2_runs", "open": "dbf2_runs_open",
            "msg": "dbf2_runs_msg"}[a.dataset]
    arms = sorted({x for _, hi, lo, _ in CONTRASTS for x in (hi, lo)})

    # Per-seed probabilities, kept SEPARATE.
    #
    # This is the subtle part and an earlier version of this script got it wrong.
    # The reported contrasts average each target's average precision ACROSS SEEDS
    # and then difference. Averaging the probabilities across seeds first and
    # computing one AP is a different quantity - it is a three-seed ensemble, which
    # is a better model than any single seed and improves the arms unevenly. Doing
    # that inflated M1R-M0 from the reported +0.0149 to +0.0327 before any
    # resampling took place. AP must therefore be computed per seed and averaged
    # per target, inside every bootstrap draw.
    prob, target, groups = {}, None, None
    for arm in arms:
        stack = []
        for suf in SEEDS.values():
            f = root / runs / f"{arm}{suf}" / f"{arm}{suf}_test_predictions.npz"
            d = np.load(f, allow_pickle=True)
            stack.append(d["prob"].astype(np.float64))
            if target is None:
                target = d["target"].astype(np.float64)
                groups = d["group_id"]
        prob[arm] = np.stack(stack)            # (n_seeds, n_molecules, n_targets)
    assert target is not None

    # scored targets: the rule the reported metrics use
    pos = target.sum(axis=0)
    scored = (pos >= a.min_positives) & (pos <= target.shape[0] - a.min_positives)
    target = target[:, scored]
    for arm in arms:
        prob[arm] = prob[arm][:, :, scored]
    print(f"  {target.shape[0]:,} held-out molecules, {int(scored.sum())} scored targets")

    # scaffold group per held-out molecule; acyclic structures fall back to their
    # own identifier, exactly as the fold assignment does
    mol = pd.read_parquet(root / prep / "molecules.parquet")
    mol = mol.set_index("mol_idx")
    scaf = mol.scaffold.reindex(groups).to_numpy()
    ikey = mol.inchikey14.reindex(groups).to_numpy()
    key = np.where(pd.isna(scaf) | (scaf == ""), ikey, scaf)
    uniq, cluster = np.unique(key, return_inverse=True)
    members = [np.flatnonzero(cluster == c) for c in range(uniq.size)]
    print(f"  {uniq.size:,} scaffold clusters "
          f"(median {int(np.median([m.size for m in members]))} molecules)")

    rng = np.random.default_rng(a.seed)
    draws = {k: np.empty(a.draws) for k, _, _, _ in CONTRASTS}
    for b in range(a.draws):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([members[c] for c in pick])
        y = target[idx]
        # Apply the SAME scoring rule the reported metric uses: a target enters the
        # macro average only if it retains at least `min_positives` positives in the
        # resample. Admitting targets with one or two positives, as an earlier
        # version did, changes the estimand - those APs are extremely high variance
        # and do not cancel between arms - and the bootstrap then fails to centre on
        # the point estimate.
        n_pos = y.sum(axis=0)
        keep = (n_pos >= a.min_positives) & (n_pos <= y.shape[0] - a.min_positives)
        yk = y[:, keep]
        # AP per seed, then mean over seeds per target, then mean over targets
        macro = {arm: float(np.nanmean(
                    np.mean([ap_matrix(yk, prob[arm][si][idx][:, keep])
                             for si in range(prob[arm].shape[0])], axis=0)))
                 for arm in arms}
        for name, hi, lo, _ in CONTRASTS:
            draws[name][b] = macro[hi] - macro[lo]
        if (b + 1) % 200 == 0:
            print(f"\r  {b + 1}/{a.draws} draws", end="", flush=True)
    print()

    # centring check: the bootstrap mean must sit near the point estimate, or the
    # resample is not estimating the same quantity as the reported metric
    macro_full = {arm: float(np.nanmean(
                     np.mean([ap_matrix(target, prob[arm][si])
                              for si in range(prob[arm].shape[0])], axis=0)))
                  for arm in arms}
    print("\n  centring check (bootstrap mean vs point estimate):")
    worst = 0.0
    for name, hi, lo, _ in CONTRASTS:
        point = macro_full[hi] - macro_full[lo]
        gap = abs(draws[name].mean() - point)
        worst = max(worst, gap)
        flag = "" if gap < 0.25 * max(abs(point), 1e-6) else "   <-- OFF-CENTRE"
        print(f"    {name:11s} point {point:+.4f}  boot {draws[name].mean():+.4f}"
              f"  gap {gap:+.4f}{flag}")
    if worst > 0.01:
        print("\n  WARNING: bootstrap does not centre on the point estimates; "
              "do not report these intervals.")

    out = {"dataset": a.dataset, "draws": a.draws, "resampling_unit": "scaffold cluster",
           "n_clusters": int(uniq.size), "n_molecules": int(target.shape[0]),
           "n_scored_targets": int(scored.sum()), "contrasts": {}}
    print("\n  contrast     kind        mean       95% scaffold-cluster interval")
    for name, _, _, kind in CONTRASTS:
        d = draws[name]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["contrasts"][name] = {"kind": kind, "mean": float(d.mean()),
                                  "sd": float(d.std(ddof=1)),
                                  "lo": float(lo), "hi": float(hi)}
        print(f"  {name:11s}  {kind:10s}  {d.mean():+.4f}   [{lo:+.4f}, {hi:+.4f}]")

    dest = root / runs / "scaffold_cluster_bootstrap.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {dest}")


if __name__ == "__main__":
    main()
