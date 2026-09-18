"""Seed replication and bin-width sweep analysis for the representation paper."""
import json, itertools, pathlib
import numpy as np, pandas as pd

R = pathlib.Path("dbf2_runs")
SEEDS = {"2026": "", "7": "_s7", "13": "_s13"}
RUNGS = ["M0", "M1R", "M1", "M2"]
WIDTHS = [("0.5", "M1R"), ("0.25", "W0.25"), ("0.1", "W0.1"),
          ("0.05", "W0.05"), ("0.01", "W0.01"), ("0", "M1")]

def rep(tag):
    return json.load(open(R / tag / f"{tag}_report.json"))["test"]["macro_auprc"]

def preds(tag):
    return np.load(R / tag / f"{tag}_test_predictions.npz")

def ap(y, p):
    n = int(y.sum())
    if n == 0 or n == y.size: return np.nan
    o = np.argsort(-p, kind="stable"); y = y[o]
    return float((np.cumsum(y) / np.arange(1, y.size + 1) * y).sum() / n)

out = {}

# ---------------------------------------------------------------- 1. seed spread
tab = {}
for r in RUNGS:
    vals = [rep(r + suf) for suf in SEEDS.values()]
    tab[r] = {"values": vals, "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
out["by_rung"] = tab
print("=== macro AUPRC across 3 seeds ===")
for r in RUNGS:
    t = tab[r]
    print(f"  {r:4s} {t['mean']:.4f} +/- {t['sd']:.4f}   ({', '.join(f'{v:.4f}' for v in t['values'])})")

# ---------------------------------------------------------------- 2. per-seed effects
EFF = [("encoder", "M0", "M1R"), ("mass axis", "M1R", "M1"), ("aggregation", "M1", "M2")]
per_seed = {n: [] for n, _, _ in EFF}
for suf in SEEDS.values():
    for n, lo, hi in EFF:
        per_seed[n].append(rep(hi + suf) - rep(lo + suf))
out["effects"] = {n: {"values": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                  for n, v in per_seed.items()}
print("\n=== effect sizes, per seed ===")
for n in per_seed:
    v = per_seed[n]
    print(f"  {n:12s} {np.mean(v):+.4f} +/- {np.std(v, ddof=1):.4f}   ({', '.join(f'{x:+.4f}' for x in v)})")

# ---------------------------------------------------------------- 3. ordering robustness
print("\n=== is the ordering stable across seeds? ===")
order = {}
for a, b in itertools.combinations([n for n, _, _ in EFF], 2):
    d = np.array(per_seed[a]) - np.array(per_seed[b])
    agree = int((d > 0).sum()); n_seeds = len(d)
    order[f"{a} - {b}"] = {"diffs": d.tolist(), "mean": float(d.mean()),
                           "seeds_positive": agree, "n_seeds": n_seeds,
                           "consistent": bool(agree in (0, n_seeds))}
    verdict = "CONSISTENT" if agree in (0, n_seeds) else "REVERSES"
    print(f"  {a:12s} - {b:12s} = {d.mean():+.4f}  "
          f"[{', '.join(f'{x:+.4f}' for x in d)}]  {agree}/{n_seeds} positive  {verdict}")
out["ordering"] = order

# ------------------------------------------------- 4. paired per-target, seed-averaged
y = preds("M1")["target"]
scored = [j for j in range(y.shape[1]) if 5 <= y[:, j].sum() <= y.shape[0] - 5]
apm = {}
for r in RUNGS:
    stack = []
    for suf in SEEDS.values():
        p = preds(r + suf)["prob"]
        stack.append(np.array([ap(y[:, j].astype(float), p[:, j]) for j in scored]))
    apm[r] = np.mean(stack, axis=0)          # average AP over seeds, per target
rng = np.random.RandomState(0)
paired = {}
print("\n=== paired per-target differences (AP averaged over 3 seeds) ===")
for n, lo, hi in EFF + [("encoder+axis", "M0", "M1"), ("all three", "M0", "M2")]:
    d = apm[hi] - apm[lo]
    idx = rng.randint(0, len(d), (5000, len(d)))
    dr = d[idx].mean(1)
    paired[f"{hi}-{lo}"] = {"label": n, "mean": float(d.mean()),
                            "lo": float(np.quantile(dr, .025)),
                            "hi": float(np.quantile(dr, .975)),
                            "frac_improved": float((d > 0).mean())}
    v = paired[f"{hi}-{lo}"]
    print(f"  {hi:4s} - {lo:4s} {n:13s} {v['mean']:+.4f} "
          f"[{v['lo']:+.4f}, {v['hi']:+.4f}]  {v['frac_improved']:.1%} of targets")
out["paired"] = paired
out["chance"] = float(y[:, scored].mean()); out["n_scored"] = len(scored)

# ---------------------------------------------------------------- 5. bin-width sweep
print("\n=== bin-width sweep (seed 2026 only) ===")
base, full = rep("M1R"), rep("M1")
sweep = []
for w, tag in WIDTHS:
    v = rep(tag)
    frac = (v - base) / (full - base)
    sweep.append({"width_da": float(w), "tag": tag, "macro_auprc": v,
                  "frac_of_precision_gain": float(frac)})
    print(f"  {w:>5s} Da  ({tag:5s})  AUPRC={v:.4f}   recovers {frac:6.1%} of the 0.5 Da -> full gap")
out["sweep"] = sweep

# ---------------------------------------------- 6. absorption at each width (mechanism)
t = pd.read_parquet("data/test.parquet",
                    columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"])
cleaned = []
for a, v, p in zip(t.ms2_mzs, t.ms2_normalized_intensities, t.precursor_mz):
    a = np.asarray(a, float); v = np.asarray(v, float)
    if v.size == 0: continue
    k = (v >= 1e-3 * v.max()) & (a < float(p) + 1.5)
    if k.any(): cleaned.append(a[k])
print("\n=== peaks absorbed at each width (cleaned spectra) ===")
absorb = {}
for w, _ in WIDTHS:
    wf = float(w)
    if wf == 0: absorb[w] = 0.0; print(f"  {w:>5s} Da   0.0%"); continue
    tot = kept = 0
    for a in cleaned:
        b = np.floor(a / wf).astype(np.int64)
        tot += b.size; kept += np.unique(b).size
    absorb[w] = float(1 - kept / tot)
    print(f"  {w:>5s} Da   {100*(1-kept/tot):5.1f}%")
out["absorption_by_width"] = absorb

json.dump(out, open("dbf2_runs/seed_sweep_analysis.json", "w"), indent=1)
print("\nwrote dbf2_runs/seed_sweep_analysis.json")
