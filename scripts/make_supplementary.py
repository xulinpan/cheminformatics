"""Build the supplementary data tables: per-arm metrics, per-target average
precision, per-seed effects and bootstrap distributions, for both corpora."""
import json, pathlib, numpy as np, pandas as pd

OUT = pathlib.Path("cheminformatics/results")
SEEDS = {"2026": "", "7": "_s7", "13": "_s13"}
RUNGS = ["M0", "M1D", "M1R", "M1", "M2"]
CORPORA = {"open": pathlib.Path("dbf2_runs_open"), "single_library": pathlib.Path("dbf2_runs")}


def ap_matrix(run_dir, tag, scored=None):
    d = np.load(run_dir / tag / f"{tag}_test_predictions.npz")
    y = d["target"].astype(np.float64); p = d["prob"].astype(np.float64)
    if scored is None:
        pos = y.sum(0); scored = np.where((pos >= 5) & (pos <= y.shape[0] - 5))[0]
    y, p = y[:, scored], p[:, scored]
    o = np.argsort(-p, axis=0, kind="stable")
    ys = np.take_along_axis(y, o, axis=0)
    prec = np.cumsum(ys, axis=0) / np.arange(1, y.shape[0] + 1)[:, None]
    return (prec * ys).sum(0) / ys.sum(0), scored


def metrics_table(run_dir, corpus):
    rows = []
    for r in RUNGS:
        for seed, suf in SEEDS.items():
            f = run_dir / (r + suf) / f"{r+suf}_report.json"
            if not f.exists():
                continue
            d = json.load(open(f)); t = d["test"]
            rows.append({"corpus": corpus, "arm": r, "seed": seed,
                         "parameters": d["n_parameters"], "best_epoch": d.get("best_epoch"),
                         "macro_auprc": t["macro_auprc"], "macro_auroc": t["macro_auroc"],
                         "micro_auprc": t.get("micro_auprc"), "brier": t.get("brier"),
                         "n_molecules": t["n_molecules"]})
    return pd.DataFrame(rows)


all_metrics, all_ap, all_boot = [], [], []
for corpus, run_dir in CORPORA.items():
    all_metrics.append(metrics_table(run_dir, corpus))
    aps, scored = {}, None
    for r in RUNGS:
        cols = []
        for seed, suf in SEEDS.items():
            if not (run_dir / (r + suf)).exists():
                continue
            a, scored = ap_matrix(run_dir, r + suf, scored)
            cols.append(a)
        if not cols:
            continue
        aps[r] = np.mean(cols, axis=0)
        for seed, col in zip(SEEDS, cols):
            aps[f"{r}_seed{seed}"] = col
    df = pd.DataFrame(aps); df.insert(0, "target_index", scored); df.insert(0, "corpus", corpus)
    all_ap.append(df)

    rng = np.random.RandomState(0)
    for name, lo, hi in (("interface", "M0", "M1D"), ("attention", "M1D", "M1R"),
                         ("encoder", "M0", "M1R"), ("mass axis", "M1R", "M1"),
                         ("aggregation", "M1", "M2")):
        if lo not in aps or hi not in aps:
            continue
        d = aps[hi] - aps[lo]
        draws = d[rng.randint(0, len(d), (5000, len(d)))].mean(1)
        all_boot.append(pd.DataFrame({"corpus": corpus, "effect": name, "draw": draws}))

pd.concat(all_metrics).to_csv(OUT / "si_table_s1_arm_metrics.csv", index=False)
pd.concat(all_ap).to_csv(OUT / "si_table_s2_per_target_ap.csv", index=False)
boot = pd.concat(all_boot)
boot.to_csv(OUT / "si_table_s3_bootstrap_draws.csv.gz", index=False, compression="gzip")
summ = (boot.groupby(["corpus", "effect"]).draw
        .agg(mean="mean", sd="std",
             q025=lambda x: x.quantile(.025), q975=lambda x: x.quantile(.975))
        .round(5).reset_index())
summ.to_csv(OUT / "si_table_s3_bootstrap_summary.csv", index=False)
print(pd.concat(all_metrics).to_string(index=False))
print()
print(summ.to_string(index=False))
print(f"\nper-target AP rows: {len(pd.concat(all_ap)):,}")
