"""Figures for the seed-replicated decomposition and the bin-width sweep."""
import json, pathlib, importlib.util
import numpy as np
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("figs", "dbf2/figures.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
F.style()
A = json.load(open("dbf2_runs/seed_sweep_analysis.json"))
out = pathlib.Path("figures")

# ------------------------------------------------------------ fig3: decomposition
RUNGS = ["M0", "M1R", "M1", "M2"]
LAB = {"M0": "M0\nbinned\nMLP", "M1R": "M1R\nbinned\nTransformer",
       "M1": "M1\nfull precision\nTransformer", "M2": "M2\nfull precision\n+ pooling"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.5))

a1.axhline(A["chance"], color=F.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
a1.text(-0.45, A["chance"] - 0.004, f"chance {A['chance']:.3f}", color=F.MUTED,
        fontsize=7.4, va="top", ha="left")
for i, r in enumerate(RUNGS):
    t = A["by_rung"][r]; c = F.SERIES[i % 3]
    a1.plot([i, i], [0, t["mean"]], color=c, lw=2.2, solid_capstyle="round", zorder=2)
    a1.plot([i] * len(t["values"]), t["values"], "o", color=c, ms=4.6, alpha=0.75,
            mec="none", zorder=5)
    a1.plot([i], [t["mean"]], "o", color=c, ms=8.5, mec=F.SURFACE, mew=1.5, zorder=4)
    a1.text(i, t["mean"] + 0.009, f"{t['mean']:.4f}", ha="center", va="bottom",
            color=F.INK, fontsize=8.4, fontweight="bold")
a1.set_xticks(range(4)); a1.set_xticklabels([LAB[r] for r in RUNGS],
                                            color=F.INK_2, fontsize=7.2)
a1.set_xlim(-0.5, 3.5); a1.set_ylim(-0.016, 0.175)
F.finish(a1, "Four arms, one change each",
         "held-out macro AUPRC; 3 seeds (small dots), mean (large)",
         ylabel="macro AUPRC")

keys = [("M1R-M0", "encoder", F.SERIES[0]), ("M1-M1R", "mass axis", F.SERIES[1]),
        ("M2-M1", "aggregation", F.SERIES[2])]
y = np.arange(3)[::-1]
a2.axvline(0, color=F.BASELINE, lw=1.0, zorder=1)
for yi, (k, name, c) in zip(y, keys):
    v = A["paired"][k]
    a2.plot([v["lo"], v["hi"]], [yi, yi], color=c, lw=2.4, solid_capstyle="round", zorder=2)
    a2.plot(A["effects"][name]["values"], [yi] * 3, "o", color=c, ms=4.6, alpha=0.75,
            mec="none", zorder=5)
    a2.plot([v["mean"]], [yi], "o", color=c, ms=8.5, mec=F.SURFACE, mew=1.5, zorder=4)
    a2.text(v["hi"] + 0.0016, yi + 0.13, f"+{v['mean']:.4f}", va="center",
            color=F.INK, fontsize=8.4, fontweight="bold")
    a2.text(v["hi"] + 0.0016, yi - 0.19, f"{v['frac_improved']:.0%} of targets",
            va="center", color=F.MUTED, fontsize=7.4)
a2.set_yticks(y); a2.set_yticklabels([f"{k.split('-')[0]} − {k.split('-')[1]}\n{n}"
                                      for k, n, _ in keys], color=F.INK_2, fontsize=7.6)
a2.set_ylim(-0.75, 2.6); a2.set_xlim(-0.002, 0.062)
F.finish(a2, "What each change is worth", "paired over targets, 95% interval",
         xlabel="Δ AUPRC", grid_axis="x")
fig.tight_layout(w_pad=2.4)
F.save(fig, out, "fig3_decomposition")

# --------------------------------------------------------------- fig4: bin-width
sw = [s for s in A["sweep"] if s["width_da"] > 0]
w = np.array([s["width_da"] for s in sw]); v = np.array([s["macro_auprc"] for s in sw])
o = np.argsort(-w); w, v = w[o], v[o]
absb = np.array([A["absorption_by_width"][f"{x:g}"] for x in w])
full, base = A["by_rung"]["M1"]["mean"], A["by_rung"]["M1R"]["mean"]
sd = A["by_rung"]["M1"]["sd"]

fig, (b1, b2) = plt.subplots(2, 1, figsize=(5.4, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.45, 1]})
b1.axhspan(full - sd, full + sd, color=F.SERIES[2], alpha=0.13, zorder=1)
b1.axhline(full, color=F.SERIES[2], lw=1.3, ls=(0, (5, 3)), zorder=2)
b1.text(0.0115, full + 0.0016, "full precision", color=F.SERIES[2], fontsize=7.6,
        fontweight="bold", va="bottom")
b1.plot(w, v, "-", color=F.SERIES[0], lw=1.9, zorder=3)
b1.plot(w, v, "o", color=F.SERIES[0], ms=7, mec=F.SURFACE, mew=1.4, zorder=4)
for x, yv in zip(w, v):
    b1.annotate(f"{yv:.4f}", (x, yv), textcoords="offset points",
                xytext=(0, 9 if x <= 0.05 else -13), ha="center",
                color=F.INK_2, fontsize=7.2)
b1.set_xscale("log"); b1.set_ylim(0.072, 0.124)
F.finish(b1, "Precision pays only below 0.05 Da",
         "one run per width; shaded band is the full-precision seed spread",
         ylabel="macro AUPRC")

b2.plot(w, 100 * absb, "-", color=F.CRIT, lw=1.9, zorder=3)
b2.plot(w, 100 * absb, "o", color=F.CRIT, ms=7, mec=F.SURFACE, mew=1.4, zorder=4)
for x, yv in zip(w, absb):
    b2.text(x, 100 * yv + 1.6, f"{100*yv:.0f}%", ha="center", va="bottom",
            color=F.INK_2, fontsize=7.2)
b2.set_xscale("log"); b2.set_ylim(0, 48)
b2.set_xticks(w); b2.set_xticklabels([f"{x:g}" for x in w])
b2.minorticks_off()
F.finish(b2, "Peaks absorbed into an occupied bin", None,
         xlabel="bin width (Da)", ylabel="peaks absorbed (%)")
fig.tight_layout(h_pad=1.9)
F.save(fig, out, "fig4_binwidth")
print("done")
