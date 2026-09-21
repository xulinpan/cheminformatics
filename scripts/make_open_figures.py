"""Figures for the open-corpus decomposition and the two-corpus contrast."""
import json, pathlib, importlib.util
import numpy as np
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("figs", "dbf2/figures.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
F.style()
OP = json.load(open("dbf2_runs_open/seed_analysis_open.json"))
SL = json.load(open("dbf2_runs/seed_sweep_analysis.json"))
RP = json.load(open("dbf2_runs_open/representation_open.json"))
RH = json.load(open("dbf2_runs/representation_heldout.json"))
out = pathlib.Path("figures")

RUNGS = ["M0", "M1D", "M1R", "M1", "M2"]
LAB = {"M0": "M0\nbinned\ndense MLP",
       "M1D": "M1D\nbinned\npeak tokens",
       "M1R": "M1R\nbinned\n+ attention",
       "M1": "M1\nfull precision\n+ attention",
       "M2": "M2\nfull precision\n+ pooling"}

# ------------------------------------------------ fig1: decomposition (open)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
a1.axhline(OP["chance"], color=F.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
a1.text(-0.45, OP["chance"] - 0.005, f"chance {OP['chance']:.3f}", color=F.MUTED,
        fontsize=7.4, va="top", ha="left")
for i, r in enumerate(RUNGS):
    t = OP["by_rung"][r]; c = F.SERIES[i % 3]
    a1.plot([i, i], [0, t["mean"]], color=c, lw=2.2, solid_capstyle="round", zorder=2)
    a1.plot([i] * len(t["values"]), t["values"], "o", color=c, ms=4.6, alpha=0.75,
            mec="none", zorder=5)
    a1.plot([i], [t["mean"]], "o", color=c, ms=8.5, mec=F.SURFACE, mew=1.5, zorder=4)
    a1.text(i, t["mean"] + 0.010, f"{t['mean']:.4f}", ha="center", va="bottom",
            color=F.INK, fontsize=8.4, fontweight="bold")
a1.set_xticks(range(len(RUNGS))); a1.set_xticklabels([LAB[r] for r in RUNGS],
                                            color=F.INK_2, fontsize=7.0)
a1.set_xlim(-0.5, len(RUNGS) - 0.5); a1.set_ylim(-0.018, 0.225)
F.finish(a1, "Five arms, one change each",
         "held-out macro AUPRC, 10,058 molecules; 3 seeds (small), mean (large)",
         ylabel="macro AUPRC")

keys = [("M1D-M0", "interface", F.SERIES[0]), ("M1R-M1D", "attention", F.MUTED),
        ("M1-M1R", "mass axis", F.SERIES[1]), ("M2-M1", "aggregation", F.SERIES[2])]
y = np.arange(len(keys))[::-1]
a2.axvline(0, color=F.BASELINE, lw=1.0, zorder=1)
for yi, (k, name, c) in zip(y, keys):
    v = OP["paired"][k]
    a2.plot([v["lo"], v["hi"]], [yi, yi], color=c, lw=2.4, solid_capstyle="round", zorder=2)
    a2.plot(OP["effects"][name]["values"], [yi] * 3, "o", color=c, ms=4.6, alpha=0.75,
            mec="none", zorder=5)
    a2.plot([v["mean"]], [yi], "o", color=c, ms=8.5, mec=F.SURFACE, mew=1.5, zorder=4)
    a2.text(max(v["hi"], 0.0) + 0.0018, yi + 0.13, f"{v['mean']:+.4f}", va="center",
            color=F.INK, fontsize=8.4, fontweight="bold")
    a2.text(max(v["hi"], 0.0) + 0.0018, yi - 0.19, f"{v['frac_improved']:.0%} of targets",
            va="center", color=F.MUTED, fontsize=7.4)
a2.set_yticks(y); a2.set_yticklabels([f"{k.split('-')[0]} − {k.split('-')[1]}\n{n}"
                                      for k, n, _ in keys], color=F.INK_2, fontsize=7.6)
a2.set_ylim(-0.75, len(keys) - 0.4); a2.set_xlim(-0.012, 0.075)
F.finish(a2, "What each change is worth", "paired over targets, 95% interval",
         xlabel="Δ AUPRC", grid_axis="x")
fig.tight_layout(w_pad=2.4)
F.save(fig, out, "fig1_decomposition_open")

# ------------------------------------------ fig2: the two corpora side by side
names = ["encoder", "mass axis", "aggregation"]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(8.6, 3.7))
h = 0.19
b1.axvline(0, color=F.BASELINE, lw=1.0, zorder=1)
for i, n in enumerate(names):
    yi = len(names) - 1 - i
    for j, (src, lbl, c) in enumerate(((SL, "single library", F.SERIES[0]),
                                       (OP, "open, 66 instruments", F.SERIES[1]))):
        e = src["effects"][n]; off = (0.5 - j) * 2 * h
        b1.plot(e["values"], [yi + off] * len(e["values"]), "o", color=c, ms=4.2,
                alpha=0.7, mec="none", zorder=4)
        b1.plot([e["mean"]], [yi + off], "o", color=c, ms=8, mec=F.SURFACE, mew=1.4, zorder=5)
        b1.text(0.0635, yi + off, f"+{e['mean']:.4f}", va="center", ha="left",
                color=c, fontsize=7.8, fontweight="bold")
b1.set_yticks(range(len(names))); b1.set_yticklabels(names[::-1], color=F.INK_2, fontsize=8.2)
b1.set_ylim(-0.6, len(names) - 0.25); b1.set_xlim(-0.002, 0.079)
from matplotlib.lines import Line2D
b1.legend(handles=[Line2D([], [], marker="o", ls="", color=F.SERIES[0], label="single library (99% timsTOF)"),
                   Line2D([], [], marker="o", ls="", color=F.SERIES[1], label="open (66 instruments)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False,
          labelcolor=F.INK_2, fontsize=7.6, handletextpad=0.4, columnspacing=1.6)
F.finish(b1, "The same decomposition on two corpora",
         "per-seed effects (small) and their mean (large)",
         xlabel="Δ AUPRC", grid_axis="x")

share = [100 * SL["paired"]["M1R-M0"]["mean"] / SL["paired"]["M1-M0"]["mean"],
         100 * OP["paired"]["M1R-M0"]["mean"] / OP["paired"]["M1-M0"]["mean"]]
xs = np.arange(2)
for i, (v, c) in enumerate(zip(share, [F.SERIES[0], F.SERIES[1]])):
    b2.plot([i, i], [0, v], color=c, lw=2.4, solid_capstyle="round", zorder=2)
    b2.plot([i], [v], "o", color=c, ms=9, mec=F.SURFACE, mew=1.5, zorder=3)
    b2.text(i, v + 2.2, f"{v:.0f}%", ha="center", color=F.INK, fontsize=9, fontweight="bold")
b2.axhline(50, color=F.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
b2.text(-0.42, 51.5, "half", color=F.MUTED, fontsize=7.4, va="bottom")
b2.set_xticks(xs); b2.set_xticklabels(["single\nlibrary", "open\n66 instruments"],
                                      color=F.INK_2, fontsize=8)
b2.set_xlim(-0.5, 1.5); b2.set_ylim(0, 92)
F.finish(b2, "Share of the gain that is the encoder",
         "of what a two-arm comparison credits to representation",
         ylabel="% of M1 − M0")
fig.tight_layout(w_pad=2.4, rect=(0, 0.06, 1, 1))
F.save(fig, out, "fig2_two_corpora")

# ------------------------------------------- fig3: absorption vs width, both
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for src, lbl, c in ((RH, "single library", F.SERIES[0]), (RP, "open", F.SERIES[1])):
    a = src["absorption_by_width"]
    w = sorted((float(k) for k in a if float(k) > 0), reverse=True)
    v = [100 * a[f"{x:g}"] for x in w]
    ax.plot(w, v, "-", color=c, lw=1.9, zorder=3)
    ax.plot(w, v, "o", color=c, ms=6, mec=F.SURFACE, mew=1.3, zorder=4)
    ax.annotate(lbl, (w[0], v[0]), textcoords="offset points", xytext=(-4, 7),
                ha="right", color=c, fontsize=8, fontweight="bold")
ax.set_xscale("log"); ax.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5])
ax.set_xticklabels(["0.01", "0.05", "0.1", "0.25", "0.5"]); ax.minorticks_off()
ax.set_xlim(0.008, 0.62); ax.set_ylim(0, 42)
F.finish(ax, "Peaks absorbed into an occupied bin",
         "held-out spectra of each corpus", xlabel="bin width (Da)",
         ylabel="peaks absorbed (%)")
fig.tight_layout()
F.save(fig, out, "fig3_absorption_both")
print("done")
