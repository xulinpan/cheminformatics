"""Figures for the main manuscript (paperA).

Three figures, all from saved analysis artefacts:
  figure1_arm_means      five-arm ladder, per-seed points over the three-seed mean
  figure2_contrasts      paired target-level contrasts with target-resampling intervals
  figure3_absorption     peak absorption against grid width, both corpora

Per-seed points are on figure 1 deliberately: the paper's central claim is that two
increments cannot be ordered because their difference changes sign across seeds, so
the seeds belong on the figure rather than only in a table.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cheminformatics" / "manuscript"
OPEN = json.load(open(ROOT / "dbf2_runs_open" / "seed_analysis_open.json"))
ABS_OPEN = json.load(open(ROOT / "dbf2_runs_open" / "representation_open.json"))
ABS_SL = json.load(open(ROOT / "dbf2_runs" / "representation_heldout.json"))

BLUE, GREY, RED = "#2f5c9e", "#6b7280", "#a33b3b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


# ---------------------------------------------------------------- figure 1
ARMS = ["M0", "M1D", "M1R", "M1", "M2"]
LABEL = {"M0": "M0\nbinned MLP\n+ metadata",
         "M1D": "M1D\nrounded tokens\nattention-free",
         "M1R": "M1R\nrounded tokens\nattention-bearing",
         "M1": "M1\nfull precision\nattention-bearing",
         "M2": "M2\nfull precision\nhierarchical"}

fig, ax = plt.subplots(figsize=(7.0, 3.5))
xs = range(len(ARMS))
means = [OPEN["by_rung"][a]["mean"] for a in ARMS]
ax.plot(xs, means, "-", color=BLUE, lw=1.4, zorder=2)
for x, a in zip(xs, ARMS):
    vals = OPEN["by_rung"][a]["values"]
    ax.plot([x] * len(vals), vals, "o", ms=3.4, color=GREY, alpha=0.75,
            zorder=3, label="individual seeds" if x == 0 else None)
    ax.plot([x], [OPEN["by_rung"][a]["mean"]], "o", ms=7.5, color=BLUE,
            zorder=4, label="three-seed mean" if x == 0 else None)
    ax.annotate(f"{OPEN['by_rung'][a]['mean']:.4f}", (x, OPEN["by_rung"][a]["mean"]),
                textcoords="offset points", xytext=(0, 11), ha="center", fontsize=8)
ax.axhline(OPEN["chance"], ls="--", lw=1, color=RED)
ax.annotate(f"chance = {OPEN['chance']:.4f}", (0, OPEN["chance"]),
            textcoords="offset points", xytext=(4, 5), fontsize=7.5, color=RED)
ax.set_xticks(list(xs))
ax.set_xticklabels([LABEL[a] for a in ARMS], fontsize=7.6)
ax.set_ylabel("macro AUPRC")
ax.set_ylim(0, max(means) * 1.16)
ax.legend(frameon=False, fontsize=7.8, loc="upper left")
save(fig, "figure1_arm_means")

# ---------------------------------------------------------------- figure 2
# Kind is encoded by marker shape as well as colour, so the distinction between a
# single-factor intervention and a composite contrast survives greyscale printing
# and colour-vision deficiency. Labels sit left of each interval rather than at the
# right-hand end, where they previously collided with the widest intervals.
ROWS = [("M1D-M0", "M1D $-$ M0", "binned token-encoder package", True),
        ("M1R-M1D", "M1R $-$ M1D", "attention-bearing vs attention-free", True),
        ("M1R-M0", "M1R $-$ M0", "rounded token-encoder package", True),
        ("M1-M1R", "M1 $-$ M1R", "mass-axis precision", False),
        ("M2-M1", "M2 $-$ M1", "multi-spectrum aggregation", False)]

fig, ax = plt.subplots(figsize=(7.4, 3.4))
for i, (key, name, interp, composite) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    p = OPEN["paired"][key]
    colour = GREY if composite else BLUE
    marker = "s" if composite else "o"
    ax.plot([p["lo"], p["hi"]], [y, y], "-", color=colour, lw=1.6, zorder=2)
    for x in (p["lo"], p["hi"]):                      # interval caps aid reading
        ax.plot([x, x], [y - 0.12, y + 0.12], "-", color=colour, lw=1.2, zorder=2)
    ax.plot([p["mean"]], [y], marker, ms=7 if composite else 7.5, color=colour,
            zorder=3, markeredgecolor="white", markeredgewidth=0.8)
    ax.annotate(f"{p['mean']:+.4f}", (p["mean"], y), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8, color=colour)
ax.axvline(0, ls="--", lw=1, color=RED, zorder=1)
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([f"{n}\n{i}" for _, n, i, _ in ROWS][::-1], fontsize=7.8)
ax.set_xlabel("change in macro AUPRC (paired over targets, 95% target-resampling interval)")
ax.set_xlim(-0.010, 0.030)
ax.set_ylim(-0.6, len(ROWS) - 0.25)
ax.plot([], [], "o", color=BLUE, ms=7, markeredgecolor="white",
        label="one-factor intervention")
ax.plot([], [], "s", color=GREY, ms=7, markeredgecolor="white",
        label="composite contrast")
ax.legend(frameon=False, fontsize=8, loc="lower right")
save(fig, "figure2_contrasts")

# ---------------------------------------------------------------- figure 3
fig, ax = plt.subplots(figsize=(5.0, 3.2))
for src, name, colour, mark in ((ABS_SL, "single library (99% timsTOF)", GREY, "s"),
                                (ABS_OPEN, "open (66 instrument types)", BLUE, "o")):
    w = sorted((float(k) for k in src["absorption_by_width"] if float(k) > 0), reverse=True)
    y = [100 * src["absorption_by_width"][f"{x:g}"] for x in w]
    ax.plot(w, y, f"-{mark}", color=colour, ms=4.5, lw=1.4, label=name)
ax.set_xscale("log")
ax.set_xlabel("grid width (Da)")
ax.set_ylabel("peaks absorbed into an occupied bin (%)")
ax.set_xticks([0.01, 0.05, 0.1, 0.25, 0.5])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.legend(frameon=False, fontsize=8)
save(fig, "figure3_absorption")
print("done")
