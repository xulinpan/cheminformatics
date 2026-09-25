"""Uncertainty for the contrasts against the TUNED dense baseline.

Sect. 6 reports that every peak-token arm sits below the tuned baseline, with a
seed standard deviation and nothing else. Every other contrast in the paper
carries two intervals; this one carried none, and it is the claim in the title.

Nothing here requires retraining. The tuned baseline's fold-0 predictions are
saved for all three seeds, so both intervals the paper already defines can be
computed from what is on disk:

  (ii) target-resampling  - 5,000 draws over the scored targets, after averaging
       each target's average precision across the three seeds, as in
       scripts/seed_sweep_analysis.py;
  (iii) scaffold-cluster  - draws over the scaffold clusters partitioning the
       held-out structures, macro AUPRC recomputed per arm inside each draw and
       the contrast formed inside the draw, as in
       scripts/scaffold_cluster_bootstrap.py.

Both estimators are copied from those scripts unchanged, including the rule that
a target enters the macro average only while it retains at least `min_positives`
positives in the resample. The centring check is retained: if the bootstrap mean
does not sit near the point estimate the resample is not estimating the reported
quantity and the interval must not be reported.

The scaffold bootstrap costs about a core-second per draw per arm-seed - roughly
seven core-hours for 2,000 draws over six arms - so it is split into parts. Each
part is an independent slice with its own seed, checkpointed as it goes, and a
final --merge pass combines them:

    python scripts/tuned_baseline_intervals.py --root . --part p1 --seed 1 --draws 500
    ... p2/2, p3/3, p4/4, in parallel ...
    python scripts/tuned_baseline_intervals.py --root . --merge

A part that is interrupted resumes from its checkpoint when rerun with the same
--part and --seed. Running with neither --part nor --merge does the whole thing
in one process.
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
# M1RT is the peak-token arm after its own matched search, so M1RT - M0T is the
# tuned-against-tuned comparison: both sides searched over the same grid, on the
# same folds, with fold 0 consulted by neither. It is the contrast that settles
# Sect. 6. M1RT - M1R measures what the matched search was worth.
CONTRASTS = [("M0T-M0",   "M0T",  "M0",   "comparator tuning"),
             ("M1RT-M1R", "M1RT", "M1R",  "treatment tuning"),
             ("M1RT-M0T", "M1RT", "M0T",  "tuned against tuned"),
             ("M1D-M0T",  "M1D",  "M0T",  "composite"),
             ("M1R-M0T",  "M1R",  "M0T",  "composite"),
             ("M1-M0T",   "M1",   "M0T",  "composite"),
             ("M2-M0T",   "M2",   "M0T",  "composite")]
NAMES = [k for k, _, _, _ in CONTRASTS]


def ap_matrix(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Average precision per column, vectorised, matching dbf2.evaluate."""
    order = np.argsort(-p, axis=0, kind="stable")
    ys = np.take_along_axis(y, order, axis=0)
    tp = np.cumsum(ys, axis=0)
    prec = tp / np.arange(1, y.shape[0] + 1, dtype=np.float64)[:, None]
    n_pos = ys.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (prec * ys).sum(axis=0) / n_pos


def macro(arms, prob, y, idx=None, keep=None):
    """Macro AUPRC per arm: AP per seed, mean over seeds per target, then mean."""
    out = {}
    for arm in arms:
        pr = prob[arm]
        per_seed = [ap_matrix(y, pr[si] if idx is None else pr[si][idx][:, keep])
                    for si in range(pr.shape[0])]
        out[arm] = float(np.nanmean(np.mean(per_seed, axis=0)))
    return out


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--root", default=".")
    ap_.add_argument("--dataset", default="open", choices=["default", "open", "msg"])
    ap_.add_argument("--draws", type=int, default=2000,
                     help="scaffold-cluster draws for this process")
    ap_.add_argument("--target-draws", type=int, default=5000)
    ap_.add_argument("--seed", type=int, default=2026)
    ap_.add_argument("--min-positives", type=int, default=5)
    ap_.add_argument("--part", default=None,
                     help="name this process's slice; its draws go to "
                          "results/parts/<part>.npz and nothing is summarised")
    ap_.add_argument("--merge", action="store_true",
                     help="combine every results/parts/*.npz into the final JSON")
    ap_.add_argument("--ckpt-every", type=int, default=100)
    a = ap_.parse_args()

    root = Path(a.root)
    prep = {"default": "dbf2_prepared", "open": "dbf2_prepared_open",
            "msg": "dbf2_prepared_msg"}[a.dataset]
    runs = {"default": "dbf2_runs", "open": "dbf2_runs_open",
            "msg": "dbf2_runs_msg"}[a.dataset]
    arms = sorted({x for _, hi, lo, _ in CONTRASTS for x in (hi, lo)})
    parts_dir = root / "results" / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

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
        stack = [np.load(root / runs / f"{arm}{suf}" / f"{arm}{suf}_test_predictions.npz",
                         allow_pickle=True)["prob"].astype(np.float64)[:, scored]
                 for suf in SEEDS.values()]
        prob[arm] = np.stack(stack)            # (n_seeds, n_molecules, n_scored)
    print(f"    loaded {len(arms)} arms x {prob[arms[0]].shape[0]} seeds")

    apm = {arm: np.mean([ap_matrix(target, prob[arm][si])
                         for si in range(prob[arm].shape[0])], axis=0)
           for arm in arms}
    macro_full = {arm: float(np.nanmean(apm[arm])) for arm in arms}
    print("\n  point estimates (macro AUPRC, 3-seed mean of per-target AP):")
    for arm in arms:
        print(f"    {arm:4s} {macro_full[arm]:.4f}")

    # scaffold group per held-out molecule; acyclic structures fall back to their
    # own identifier, exactly as the fold assignment does
    mol = pd.read_parquet(root / prep / "molecules.parquet").set_index("mol_idx")
    scaf = mol.scaffold.reindex(groups).to_numpy()
    ikey = mol.inchikey14.reindex(groups).to_numpy()
    key = np.where(pd.isna(scaf) | (scaf == ""), ikey, scaf)
    uniq, cluster = np.unique(key, return_inverse=True)
    members = [np.flatnonzero(cluster == c) for c in range(uniq.size)]

    # ---------------------------------------------------- scaffold-cluster draws
    if a.merge:
        files = sorted(parts_dir.glob("*.npz"))
        if not files:
            raise SystemExit(f"no parts in {parts_dir}; run with --part first")
        acc, total = {k: [] for k in NAMES}, 0
        for f in files:
            d = np.load(f, allow_pickle=True)
            n_done = int(d["n_done"])
            if n_done == 0:
                continue
            for k in NAMES:
                acc[k].append(d[k][:n_done])
            total += n_done
            print(f"    {f.name}: {n_done:,} draws")
        draws = {k: np.concatenate(v) for k, v in acc.items()}
        n_draws = total
        print(f"  merged {total:,} draws from {len(files)} part(s)")
    else:
        ckpt = parts_dir / f"{a.part or 'main'}.npz"
        n_draws = a.draws
        draws = {k: np.empty(n_draws) for k in NAMES}
        rng = np.random.default_rng(a.seed)
        start = 0
        if ckpt.exists():
            d = np.load(ckpt, allow_pickle=True)
            if int(d["target_draws"]) == n_draws and int(d["seed"]) == a.seed:
                start = int(d["n_done"])
                for k in NAMES:
                    draws[k][:start] = d[k][:start]
                rng.bit_generator.state = d["rng_state"].item()
                print(f"  resuming {ckpt.name} at draw {start:,}")

        def save(n_done):
            np.savez(ckpt, n_done=n_done, target_draws=n_draws, seed=a.seed,
                     rng_state=np.array(rng.bit_generator.state, dtype=object),
                     **{k: draws[k] for k in NAMES})

        print(f"\n  {uniq.size:,} scaffold clusters; draws {start:,}..{n_draws:,}"
              f" (seed {a.seed})")
        for b in range(start, n_draws):
            pick = rng.integers(0, uniq.size, uniq.size)
            idx = np.concatenate([members[c] for c in pick])
            y = target[idx]
            n_pos = y.sum(axis=0)
            keep = (n_pos >= a.min_positives) & (n_pos <= y.shape[0] - a.min_positives)
            m = macro(arms, prob, y[:, keep], idx, keep)
            for name, hi, lo, _ in CONTRASTS:
                draws[name][b] = m[hi] - m[lo]
            if (b + 1) % a.ckpt_every == 0:
                save(b + 1)
                print(f"    {b + 1:,}/{n_draws:,} draws", flush=True)
        save(n_draws)
        if a.part:
            print(f"\n  part complete: {ckpt}\n  rerun with --merge to summarise")
            return

    # ------------------------------------------------ (ii) target resampling
    out = {"dataset": a.dataset, "n_molecules": int(target.shape[0]),
           "n_scored_targets": int(scored.sum()), "target_draws": a.target_draws,
           "scaffold_draws": n_draws, "n_clusters": int(uniq.size),
           "macro_auprc": macro_full, "contrasts": {}}
    rng_t = np.random.default_rng(0)
    print("\n  target-resampling intervals "
          f"({a.target_draws:,} draws over {int(scored.sum())} scored targets):")
    for name, hi, lo, kind in CONTRASTS:
        d = apm[hi] - apm[lo]
        idx = rng_t.integers(0, d.size, (a.target_draws, d.size))
        dr = np.nanmean(d[idx], axis=1)
        rec = {"kind": kind, "point": float(np.nanmean(d)),
               "target_lo": float(np.quantile(dr, 0.025)),
               "target_hi": float(np.quantile(dr, 0.975)),
               "frac_improved": float(np.nanmean(d > 0))}
        out["contrasts"][name] = rec
        print(f"    {name:9s} {rec['point']:+.4f}  "
              f"[{rec['target_lo']:+.4f}, {rec['target_hi']:+.4f}]  "
              f"{rec['frac_improved']:.1%} of targets")

    print("\n  per seed:")
    for name, hi, lo, _ in CONTRASTS:
        vals = [float(np.nanmean(ap_matrix(target, prob[hi][si]))) -
                float(np.nanmean(ap_matrix(target, prob[lo][si])))
                for si in range(prob[hi].shape[0])]
        out["contrasts"][name]["per_seed"] = vals
        out["contrasts"][name]["seed_sd"] = float(np.std(vals, ddof=1))
        print(f"    {name:9s} {', '.join(f'{v:+.4f}' for v in vals)}   "
              f"{sum(v > 0 for v in vals)}/{len(vals)} positive")

    # ------------------------------------------------ (iii) scaffold summary
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
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {dest}")


if __name__ == "__main__":
    main()
