"""Supplementary figures: bootstrap distributions and stratified absorption."""
import pathlib, importlib.util
import numpy as np, pandas as pd, matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("figs", "dbf2/figures.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F); F.style()
R = pathlib.Path("cheminformatics/results"); out = pathlib.Path("figures")
boot = pd.read_csv(R / "si_table_s3_bootstrap_draws.csv.gz")

EFF = ["encoder", "mass axis", "aggregation"]
COR = [("open", "open corpus, 66 instruments"), ("single_library", "single library, 99% timsTOF")]
fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.2), sharex=True)
for ax, (corpus, title) in zip(axes, COR):
    ax.axvline(0, color=F.BASELINE, lw=1.0, zorder=1)
    lab = []
    for i, e in enumerate(EFF):
        d = boot[(boot.corpus == corpus) & (boot.effect == e)].draw.to_numpy()
        c = F.SERIES[i % 3]
        ax.hist(d, bins=70, density=True, color=c, alpha=0.5, lw=0, zorder=2)
        ax.axvline(d.mean(), color=c, lw=1.4, zorder=3)
        lab.append((d.mean(), e, c))
    # stagger the labels vertically so adjacent distributions do not collide
    top = ax.get_ylim()[1]
    for k, (x, e, c) in enumerate(sorted(lab)):
        ax.text(x, top * (1.00 - 0.135 * k), e, color=c, fontsize=7.6,
                fontweight="bold", ha="center", va="top",
                bbox=dict(fc=F.SURFACE, ec="none", pad=1.2), zorder=6)
    ax.set_ylim(0, top * 1.06)
    F.finish(ax, title, "5,000 bootstrap draws of the paired per-target mean",
             ylabel="density", grid_axis="x")
axes[-1].set_xlabel("Δ AUPRC")
axes[0].set_xlim(-0.002, 0.062)
fig.tight_layout(h_pad=1.8)
F.save(fig, out, "figS1_bootstrap_distributions")

rows = []
for name, label in (("adduct", "adduct"), ("collision_energy", "collision energy (eV)"),
                    ("instrument_family", "instrument family")):
    d = pd.read_csv(R / f"si_absorption_by_{name}.csv", index_col=0)
    d = d[d.peaks >= 20000].sort_values("peaks", ascending=False).head(8)
    if name == "collision_energy":       # ordinal axis: order by energy, not volume
        order = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-75", "75+"]
        d = d.reindex([x for x in order if x in d.index])
    rows.append((label, d))
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
for ax, (label, d) in zip(axes, rows):
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.absorbed_pct.to_numpy(), color=F.SERIES[0], height=0.62, zorder=2)
    for yi, v in zip(y, d.absorbed_pct.to_numpy()):
        ax.text(v + 0.6, yi, f"{v:.0f}%", va="center", color=F.INK_2, fontsize=7.2)
    ax.set_yticks(y); ax.set_yticklabels([str(i) for i in d.index], fontsize=7.2, color=F.INK_2)
    ax.set_xlim(0, max(35, d.absorbed_pct.max() * 1.22)); ax.set_ylim(-0.7, len(d) - 0.3)
    F.finish(ax, label, None, xlabel="peaks absorbed (%)", grid_axis="x")
fig.suptitle("Absorption at 0.5 Da by acquisition stratum, open corpus held-out spectra",
             color=F.INK, fontweight="bold", x=0.012, ha="left", y=0.995, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.0)
F.save(fig, out, "figS2_absorption_strata")
print("done")
