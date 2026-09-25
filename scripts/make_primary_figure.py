"""Primary figure: what the comparator corrections do, and what they do not.

The five-arm ladder is no longer the paper's headline, so it is no longer the
first figure. This one carries the sequence the paper argues from: the dense
comparator moves twice, under interventions that never touch the treatment, and
the peak-token arm does not move when it receives the same search budget. The
right-hand panel shows the one contrast in which a single component changes and
no comparator enters at all.

Colours are checked for categorical separation under normal and deficient colour
vision; marker shape carries identity as well, so the figure survives greyscale.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cheminformatics" / "manuscript"

BLUE, ORANGE, INK, MUTED = "#2f5c9e", "#b45309", "#1f2937", "#6b7280"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})

# every value is a measured three-seed mean reported in the manuscript
DENSE = [0.1130, 0.1418, 0.2601]      # unconditioned -> metadata matched -> searched
TOKEN = [0.1567, 0.1567, 0.1573]      # M1R; unchanged at step 1, it was already conditioned
STEPS = ["as originally\nbuilt",
         "+ matched\nacquisition metadata",
         "+ architecture-specific\nsearch"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.6, 3.5),
                             gridspec_kw={"width_ratios": [2.45, 1.0]})

x = range(3)
ax.plot(x, DENSE, "-o", color=BLUE, lw=2, ms=7, markeredgecolor="white",
        markeredgewidth=0.9, zorder=3, label="binned comparator (M0)")
ax.plot(x, TOKEN, "-s", color=ORANGE, lw=2, ms=6.5, markeredgecolor="white",
        markeredgewidth=0.9, zorder=3, label="peak-token arm (M1R)")

# labels are placed away from the crossing point rather than on it
for i, v in enumerate(DENSE):
    dy = 11 if i == 2 else -17
    ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, dy),
                ha="center", fontsize=8.5, color=INK)
for i, v in enumerate(TOKEN):
    if i == 1:
        continue
    ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=8.5, color=INK)
ax.annotate("already conditioned;\nunchanged by this step", (1, TOKEN[1]),
            textcoords="offset points", xytext=(0, 14), ha="center",
            fontsize=7.4, color=MUTED)

ax.set_xticks(list(x)); ax.set_xticklabels(STEPS, fontsize=8)
ax.set_ylabel("macro AUPRC on held-out structures")
ax.set_ylim(0.09, 0.295)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_title("Corrections to the comparator", fontsize=9.5, color=INK, pad=10)

# right panel: the within-encoder control, no comparator involved
bx.plot([0, 1], [0.1567, 0.1709], "-o", color=BLUE, lw=2, ms=7,
        markeredgecolor="white", markeredgewidth=0.9, zorder=3)
for i, (v, lab) in enumerate([(0.1567, "M1R\n0.5 Da grid"), (0.1709, "M1\nfull precision")]):
    bx.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8.5, color=INK)
bx.set_xticks([0, 1]); bx.set_xticklabels(["M1R\n0.5 Da grid", "M1\nfull precision"], fontsize=8)
bx.set_xlim(-0.45, 1.45)
bx.annotate("+0.0142", (0.5, 0.1440), ha="center", fontsize=9.5, color=BLUE)
bx.annotate("same scale as the left panel", (0.5, 0.0985), ha="center",
            fontsize=7.4, color=MUTED)
bx.set_ylim(0.09, 0.295); bx.tick_params(labelleft=False)
bx.set_title("Mass axis inside a fixed encoder", fontsize=9.5, color=INK, pad=10)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figure_comparator.{ext}", bbox_inches="tight")
print(f"  wrote {OUT/'figure_comparator.pdf'} / .png")
