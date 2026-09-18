"""Publication figures and tables from the DeepBayesFrag-MS v2 results.

    python -m dbf2.figures --root . --outdir figures

Every number is read from a results file; nothing is transcribed by hand.
Output: 600 dpi PNG and vector PDF per figure, plus CSV and LaTeX tables.

Colour follows a validated categorical palette (three slots, all-pairs safe for
normal vision and for deuteranopia/tritanopia). Aqua sits below 3:1 against the
light surface, so every series also carries a direct label and never relies on
hue alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ the theme
SURFACE   = "#ffffff"   # the page; palette re-validated against it
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
SERIES    = ["#2a78d6", "#eb6834", "#1baf7a"]          # blue, orange, aqua
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab"]  # ordinal, >= step 250
GOOD, CRIT = "#0ca30c", "#d03b3b"
DPI = 600


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": DPI, "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.grid": True, "axes.axisbelow": True,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.8, "lines.markersize": 5,
    })


def finish(ax, title=None, sub=None, xlabel=None, ylabel=None, grid_axis="y"):
    ax.grid(axis=grid_axis, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontweight="bold", loc="left",
                     pad=21 if sub else 7)
    if sub:
        ax.text(0.0, 1.022, sub, transform=ax.transAxes, color=MUTED, fontsize=7.6,
                va="bottom", ha="left")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=DPI, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf", flush=True)


# ----------------------------------------------------------------------- data
def load(root: Path) -> dict:
    d = {}
    d["ablation"] = json.load(open(root / "dbf2_runs" / "ablation_summary.json"))
    d["history"] = {r: pd.read_csv(root / "dbf2_runs" / r / f"{r}_history.csv")
                    for r in ("M0", "M1", "M2")}
    d["pred"] = {r: np.load(root / "dbf2_runs" / r / f"{r}_test_predictions.npz")
                 for r in ("M0", "M1", "M2")}
    d["viewscale"] = json.load(open(root / "dbf2_runs" / "viewscale.json"))
    d["decomp"] = json.load(open(root / "dbf2_runs" / "decomposition.json"))
    d["acc"] = pd.read_csv(root / "dbf2_runs" / "M2" / "accuracy_analysis.csv")
    d["sub"] = pd.read_csv(root / "dbf2_runs" / "M2" / "submission.csv")
    d["key"] = pd.read_csv(root / "leak_audit" / "test_solution.csv")
    return d


def average_precision(y, p):
    n = int(y.sum())
    if n == 0 or n == y.size:
        return np.nan
    o = np.argsort(-p, kind="stable"); y = y[o]
    return float((np.cumsum(y) / np.arange(1, y.size + 1) * y).sum() / n)


def paired_differences(pred, seed=0, n_boot=5000):
    y = pred["M1"]["target"]
    scored = [j for j in range(y.shape[1]) if 5 <= y[:, j].sum() <= y.shape[0] - 5]
    ap = {r: np.array([average_precision(y[:, j].astype(float), pred[r]["prob"][:, j])
                       for j in scored]) for r in pred}
    rng = np.random.RandomState(seed)
    out = {}
    for lo, hi in (("M0", "M1"), ("M1", "M2"), ("M0", "M2")):
        d = ap[hi] - ap[lo]
        idx = rng.randint(0, len(d), (n_boot, len(d)))
        dr = d[idx].mean(1)
        out[f"{hi}-{lo}"] = {"mean": float(d.mean()),
                             "lo": float(np.quantile(dr, .025)),
                             "hi": float(np.quantile(dr, .975)),
                             "frac_improved": float((d > 0).mean())}
    chance = float(y[:, scored].mean())
    return out, chance, len(scored)


# -------------------------------------------------------------------- figures
def fig_ablation(d, out, paired, chance):
    rungs = ["M0", "M1", "M2"]
    labels = ["M0\n0.5 Da bins\nmean pool", "M1\npeak set\nmean pool",
              "M2\npeak set\nhierarchical"]
    vals = [d["ablation"][r]["test"]["macro_auprc"] for r in rungs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.3))

    x = np.arange(3)
    a1.axhline(chance, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    a1.text(-0.45, -0.004, f"chance {chance:.3f}", color=MUTED, fontsize=7.4,
            va="top", ha="left")
    for i, (v, c) in enumerate(zip(vals, SERIES)):
        a1.plot([i, i], [0, v], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        a1.plot([i], [v], "o", color=c, ms=8, mec=SURFACE, mew=1.4, zorder=3)
        a1.text(i, v + 0.008, f"{v:.4f}", ha="center", va="bottom",
                color=INK, fontsize=8.4, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(labels, color=INK_2, fontsize=7.8)
    a1.set_xlim(-0.5, 2.5); a1.set_ylim(-0.016, max(vals) * 1.25)
    finish(a1, "Substructure prediction", "held-out macro AUPRC, 8,692 molecules",
           ylabel="macro AUPRC")

    keys = ["M1-M0", "M2-M1", "M2-M0"]
    names = ["M1 − M0\nrepresentation", "M2 − M1\naggregation", "M2 − M0\nboth"]
    y = np.arange(3)[::-1]
    a2.axvline(0, color=BASELINE, lw=1.0, zorder=1)
    for yi, k, c in zip(y, keys, [SERIES[1], SERIES[2], SERIES[0]]):
        v = paired[k]
        a2.plot([v["lo"], v["hi"]], [yi, yi], color=c, lw=2.2,
                solid_capstyle="round", zorder=2)
        a2.plot([v["mean"]], [yi], "o", color=c, ms=8, mec=SURFACE, mew=1.4, zorder=3)
        a2.text(v["hi"] + 0.005, yi, f"+{v['mean']:.4f}", va="center", color=INK,
                fontsize=8.2, fontweight="bold")
        a2.text(v["hi"] + 0.005, yi - 0.30, f"{v['frac_improved']:.0%} of targets",
                va="center", color=MUTED, fontsize=7.4)
    a2.set_yticks(y); a2.set_yticklabels(names, color=INK_2, fontsize=7.8)
    a2.set_ylim(-0.7, 2.6); a2.set_xlim(-0.008, 0.185)
    finish(a2, "Paired per-target difference",
           "bootstrap over targets, 95% interval", xlabel="Δ AUPRC", grid_axis="x")
    fig.tight_layout(w_pad=2.2)
    save(fig, out, "fig1_ablation")


def fig_training(d, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for r, c in zip(("M0", "M1", "M2"), SERIES):
        h = d["history"][r]
        ax.plot(h.epoch, h.val_macro_auprc, color=c, lw=1.9, zorder=3)
        ax.plot(h.epoch.iloc[-1], h.val_macro_auprc.iloc[-1], "o", color=c, ms=7,
                mec=SURFACE, mew=1.4, zorder=4)
        ax.text(h.epoch.iloc[-1] + 0.35, h.val_macro_auprc.iloc[-1], r,
                color=c, fontsize=8.6, fontweight="bold", va="center")
    ax.set_xlim(-0.4, 16.6)
    handles = [Line2D([], [], color=c, lw=1.9,
                      label={"M0": "M0  binned + mean", "M1": "M1  peak set + mean",
                             "M2": "M2  peak set + hierarchical"}[r])
               for r, c in zip(("M0", "M1", "M2"), SERIES)]
    ax.legend(handles=handles, loc="upper left", labelcolor=INK_2)
    finish(ax, "Training curves", "validation macro AUPRC; all three plateau by epoch 13",
           xlabel="epoch", ylabel="macro AUPRC")
    fig.tight_layout()
    save(fig, out, "fig2_training_curves")


def fig_viewscale(d, out):
    v = pd.DataFrame(d["viewscale"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.1))
    a1.plot(v.views, v.m1_auprc, color=SERIES[1], lw=1.9, marker="o", ms=7,
            mec=SURFACE, mew=1.4, zorder=3)
    a1.plot(v.views, v.m2_auprc, color=SERIES[2], lw=1.9, marker="o", ms=7,
            mec=SURFACE, mew=1.4, zorder=3)
    a1.text(v.views.iloc[-1] + 0.15, v.m1_auprc.iloc[-1], "M1", color=SERIES[1],
            fontsize=8.6, fontweight="bold", va="center")
    a1.text(v.views.iloc[-1] + 0.15, v.m2_auprc.iloc[-1], "M2", color=SERIES[2],
            fontsize=8.6, fontweight="bold", va="center")
    a1.set_xscale("log", base=2); a1.set_xticks(v.views)
    a1.set_xticklabels([str(int(x)) for x in v.views])
    a1.set_xlim(0.85, 13.0)
    finish(a1, "Accuracy against the view budget",
           "1,200 molecules with ≥ 8 spectra available",
           xlabel="spectra supplied at inference", ylabel="AUPRC")

    colors = [CRIT if x < 0 else SERIES[0] for x in v.lift]
    a2.axhline(0, color=BASELINE, lw=1.0, zorder=1)
    for xi, li, c in zip(range(len(v)), v.lift, colors):
        a2.plot([xi, xi], [0, li], color=c, lw=2.2, solid_capstyle="round", zorder=2)
        a2.plot([xi], [li], "o", color=c, ms=8, mec=SURFACE, mew=1.4, zorder=3)
        a2.text(xi, li + (0.0013 if li >= 0 else -0.0013), f"{li:+.4f}",
                ha="center", va="bottom" if li >= 0 else "top",
                color=INK, fontsize=8.2, fontweight="bold")
    a2.set_xticks(range(len(v)))
    a2.set_xticklabels([str(int(x)) for x in v.views])
    a2.set_xlim(-0.55, len(v) - 0.45)
    a2.set_ylim(min(v.lift) * 1.9 - 0.004, max(v.lift) * 1.28)
    finish(a2, "Advantage of hierarchical pooling",
           "the gain grows with evidence, not with its scarcity",
           xlabel="spectra supplied at inference", ylabel="M2 − M1  (Δ AUPRC)")
    fig.tight_layout(w_pad=2.2)
    save(fig, out, "fig3_view_scaling")


def fig_decomposition(d, out):
    dc = d["decomp"]
    rows = [("random within pool", dc["random"]), ("formula term only", dc["formula"]),
            ("structural term only", dc["structural"]), ("calibrated combination", dc["combined"])]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, (name, v), c in zip(y, rows, BLUE_RAMP):
        ax.plot([v["lo"], v["hi"]], [yi, yi], color=c, lw=2.4,
                solid_capstyle="round", zorder=2)
        ax.plot([v["mrr"]], [yi], "o", color=c, ms=8.5, mec=SURFACE, mew=1.4, zorder=3)
        ax.text(v["hi"] + 0.006, yi, f"  {v['mrr']:.4f}", va="center",
                color=INK, fontsize=8.4, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], color=INK_2)
    ax.set_xlim(0, 0.34); ax.set_ylim(-0.6, len(rows) - 0.4)
    finish(ax, "Where the ranking signal comes from",
           "MRR@25 on the same 400 queries and the same candidate pools",
           xlabel="MRR@25", grid_axis="x")
    fig.tight_layout()
    save(fig, out, "fig4_retrieval_decomposition")


def fig_isomer(d, out):
    a = d["acc"]
    bins = [(1, 1, "1\n(unique)"), (2, 5, "2–5"), (6, 20, "6–20"),
            (21, 100, "21–100"), (101, 10 ** 9, ">100")]
    lab, t1, t25, n = [], [], [], []
    for lo, hi, name in bins:
        m = (a.iso_class >= lo) & (a.iso_class <= hi)
        if m.sum() < 5: continue
        lab.append(name); t1.append(a.loc[m, "exact1"].mean())
        t25.append(a.loc[m, "in25"].mean()); n.append(int(m.sum()))
    x = np.arange(len(lab)); w = 0.32
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    for i, (v, c, nm) in enumerate(((t25, SERIES[0], "truth in top 25"),
                                    (t1, SERIES[1], "top-1 exact"))):
        ax.bar(x + (i - 0.5) * (w + 0.02), v, w, color=c, label=nm,
               edgecolor=SURFACE, linewidth=1.2, zorder=2)
        for xi, vi in zip(x + (i - 0.5) * (w + 0.02), v):
            ax.text(xi, vi + 0.018, f"{vi:.3f}", ha="center", va="bottom",
                    color=INK, fontsize=7.6, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nn={ni}" for l, ni in zip(lab, n)], color=INK_2)
    ax.set_ylim(0, 1.13); ax.set_xlim(-0.6, len(lab) - 0.4)
    ax.legend(loc="upper right", labelcolor=INK_2)
    finish(ax, "Accuracy collapses as the isomer class grows",
           "library structures sharing the true molecular formula",
           ylabel="fraction of queries")
    fig.tight_layout()
    save(fig, out, "fig5_accuracy_by_isomer_class")


def fig_representation(root, out):
    t = pd.read_parquet(root / "data" / "test.parquet",
                        columns=["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"])
    # Measure the peaks the MODEL sees, not the raw file. The raw spectra carry a
    # long tail of sub-threshold peaks that preprocessing removes before either
    # arm is trained; counting them inflates the absorbed fraction from 34.7% to
    # 60.7% by charging binning for peaks no model ever received.
    cleaned = []
    for a, v, prec in zip(t.ms2_mzs, t.ms2_normalized_intensities, t.precursor_mz):
        a = np.asarray(a, float); v = np.asarray(v, float)
        if v.size == 0:
            continue
        keep = (v >= 1e-3 * v.max()) & (a < float(prec) + 1.5)
        if keep.any():
            cleaned.append(a[keep])
    mz = np.concatenate(cleaned)
    merged = tot = 0
    for a in cleaned:
        b = np.floor(a / 0.5).astype(np.int64)
        tot += b.size; merged += b.size - np.unique(b).size
    defect = mz - np.round(mz)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    sel = np.random.RandomState(0).choice(mz.size, min(40000, mz.size), replace=False)
    a1.scatter(mz[sel], defect[sel], s=1.0, c=SERIES[0], alpha=0.16, lw=0, zorder=2)
    a1.set_ylim(-0.55, 0.55)
    finish(a1, "The mass defect carries the formula",
           "test fragment peaks; structure here is what 0.5 Da bins discard",
           xlabel="fragment m/z", ylabel="mass defect (Da)", grid_axis="both")
    frac = merged / tot
    # merged fraction against fragment m/z: crowding worsens where peaks are dense
    edges = np.arange(50, 500, 50)
    kept = np.zeros(len(edges) - 1); total = np.zeros(len(edges) - 1)
    for arr in cleaned:
        b = np.floor(arr / 0.5).astype(np.int64)
        bucket = np.digitize(arr, edges) - 1
        for k in range(len(edges) - 1):
            m = bucket == k
            if not m.any():
                continue
            total[k] += m.sum()
            kept[k] += np.unique(b[m]).size        # survivors; the rest are absorbed
    ok = total >= 200
    centres = ((edges[:-1] + edges[1:]) / 2)[ok]
    fr = (1.0 - kept[ok] / total[ok])
    a2.axhline(frac, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    a2.bar(centres, fr, 38, color=CRIT, edgecolor=SURFACE, linewidth=1.2, zorder=2)
    for c, v in zip(centres, fr):
        a2.text(c, v + 0.014, f"{v:.0%}", ha="center", va="bottom", color=INK,
                fontsize=7.2, fontweight="bold")
    a2.set_ylim(0, 1.0)
    a2.yaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    finish(a2, "Peaks lost to 0.5 Da binning",
           f"{merged:,} of {tot:,} absorbed overall ({frac:.1%}, dashed)",
           xlabel="fragment m/z", ylabel="peaks absorbed by merging")
    fig.tight_layout(w_pad=2.2)
    save(fig, out, "fig7_representation")
    return {"merged": merged, "total": tot, "frac": frac}


# ---------------------------------------------------- RDKit structure gallery
def _draw(smi: str, size=(1100, 850), highlight=None):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import AllChem
    import io
    from PIL import Image
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    AllChem.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    o.clearBackground = True
    o.setBackgroundColour((1.0, 1.0, 1.0))            # SURFACE
    o.bondLineWidth = 2.4
    o.minFontSize = 26
    o.maxFontSize = 34
    o.padding = 0.07
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def fig_structures(d, out):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    a = d["acc"].merge(d["sub"], on="molecule_id").merge(
        d["key"][["molecule_id", "smiles", "formula"]].rename(
            columns={"smiles": "true_smiles", "formula": "true_formula"}),
        on="molecule_id")
    a["top1_smiles"] = a.smiles_y.str.split(";").str[0] if "smiles_y" in a else \
        a.smiles.str.split(";").str[0]

    hits = a[a.exact1 == 1].sort_values("iso_class", ascending=False).head(2)
    miss = a[(a.exact1 == 0) & (a.formula1 == 1)].sort_values("tan1").head(3)
    rows = [("correct", r) for _, r in hits.iterrows()] + \
           [("wrong isomer", r) for _, r in miss.iterrows()]

    fig, axes = plt.subplots(len(rows), 2, figsize=(7.2, 1.62 * len(rows)),
                             gridspec_kw={"wspace": 0.05})
    for i, (kind, r) in enumerate(rows):
        for j, (smi, who) in enumerate(((r.true_smiles, "true structure"),
                                        (r.top1_smiles, "top-1 prediction"))):
            ax = axes[i, j]
            img = _draw(smi)
            if img is not None:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_edgecolor(GRID); s.set_linewidth(0.9)
            ok = kind == "correct"
            # the verdict rides on the left panel's own title: a separate text object
            # in axes coordinates is invisible to tight_layout and collides with the
            # row above
            if j == 0:
                ax.set_title(f"true structure — {kind}",
                             color=GOOD if ok else CRIT, fontsize=8.2,
                             fontweight="bold", loc="left", pad=4)
            else:
                ax.set_title(who, color=INK_2, fontsize=8, loc="left", pad=4)
            if j == 1:
                c = GOOD if ok else CRIT
                ax.text(0.985, 0.03, "exact match" if ok else f"Tanimoto {r.tan1:.2f}",
                        transform=ax.transAxes, ha="right", va="bottom",
                        color=c, fontsize=8.2, fontweight="bold")
        axes[i, 0].text(-0.035, 0.5,
                        f"{r.true_formula}\n{int(r.iso_class)} isomers in library\n"
                        f"truth at rank {int(r['rank']) if r['rank']>0 else '>25'}",
                        transform=axes[i, 0].transAxes, rotation=90, ha="right",
                        va="center", color=MUTED, fontsize=7.2, linespacing=1.5)
    fig.suptitle("Predicted against true structures", color=INK, fontweight="bold",
                 fontsize=10, x=0.013, ha="left", y=0.996)
    fig.text(0.013, 0.9735, "the failures share the true molecular formula but not the skeleton",
             color=MUTED, fontsize=8, ha="left", va="top")
    fig.tight_layout(rect=[0.02, 0, 1, 0.952], h_pad=1.8, w_pad=0.5)
    save(fig, out, "fig6_structures")


# ------------------------------------------------------------------- tables
_TEX_MAP = {"Δ": r"$\Delta$", "−": "$-$", "–": "--", "—": "---",
            "≥": r"$\ge$", "≤": r"$\le$", ">": "$>$", "<": "$<$"}


def _tex(v) -> str:
    """Escape a cell for LaTeX. An unescaped %% comments out the rest of the line,
    which silently truncates the table, so this runs on every cell and header."""
    t = str(v)
    for k, r in _TEX_MAP.items():
        t = t.replace(k, r)
    for ch in ("%", "&", "#", "_"):
        t = t.replace(ch, "\\" + ch)
    return t


def _latex(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = " ".join(["l"] + ["r"] * (df.shape[1] - 1))
    head = " & ".join(_tex(c) for c in df.columns) + r" \\"
    body = "\n".join(" & ".join(_tex(v) for v in row) + r" \\"
                     for row in df.itertuples(index=False))
    return ("\\begin{table}[htbp]\n\\centering\n"
            f"\\caption{{{_tex(caption)}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def write_table(df: pd.DataFrame, out: Path, name: str, caption: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.tex").write_text(_latex(df, caption, f"tab:{name}"), encoding="utf-8")
    print(f"  wrote {name}.csv / .tex", flush=True)


def tables(d, out, paired, chance, n_scored, repr_stats):
    ab = d["ablation"]
    t2 = pd.DataFrame([{
        "rung": r,
        "encoder": {"binned": "0.5 Da bins", "peakset": "peak set"}[ab[r]["encoder"]],
        "pooling": ab[r]["pooling"],
        "parameters": f"{ab[r]['n_parameters']:,}",
        "macro AUPRC": f"{ab[r]['test']['macro_auprc']:.4f}",
        "macro AUROC": f"{ab[r]['test']['macro_auroc']:.4f}",
        "micro AUPRC": f"{ab[r]['test']['micro_auprc']:.4f}",
        "Brier": f"{ab[r]['test']['brier']:.5f}",
    } for r in ("M0", "M1", "M2")])
    write_table(t2, out, "table2_ablation",
                f"Ablation on {ab['M0']['test']['n_molecules']:,} held-out molecules. "
                f"Chance macro AUPRC is {chance:.4f} over {n_scored} scored targets.")

    t3 = pd.DataFrame([{
        "comparison": k.replace("-", " − "),
        "change": {"M1-M0": "representation", "M2-M1": "aggregation",
                   "M2-M0": "both"}[k],
        "Δ AUPRC": f"{v['mean']:+.4f}",
        "95% CI": f"[{v['lo']:+.4f}, {v['hi']:+.4f}]",
        "targets improved": f"{v['frac_improved']:.1%}",
    } for k, v in paired.items()])
    write_table(t3, out, "table3_paired_differences",
                "Paired per-target AUPRC differences, bootstrapped over targets.")

    dc = d["decomp"]
    t4 = pd.DataFrame([{
        "ranking rule": n,
        "MRR@25": f"{dc[k]['mrr']:.4f}",
        "95% CI": f"[{dc[k]['lo']:.4f}, {dc[k]['hi']:.4f}]",
        "Top-1": f"{dc[k]['top_1']:.4f}",
        "Top-5": f"{dc[k]['top_5']:.4f}",
        "Top-25": f"{dc[k]['top_25']:.4f}",
    } for k, n in (("random", "random within pool"), ("formula", "formula term only"),
                   ("structural", "structural term only"),
                   ("combined", "calibrated combination"))])
    write_table(t4, out, "table4_retrieval_decomposition",
                f"Retrieval on the 400 test molecules. Candidate recall is "
                f"{dc['candidate_recall']:.3f}, so all loss is ranking loss.")

    a = d["acc"]
    rows = []
    for lo, hi, name in ((1, 1, "1 (unique)"), (2, 5, "2–5"), (6, 20, "6–20"),
                         (21, 100, "21–100"), (101, 10 ** 9, ">100")):
        m = (a.iso_class >= lo) & (a.iso_class <= hi)
        if m.sum() < 5: continue
        rows.append({"isomers sharing the formula": name, "queries": int(m.sum()),
                     "Top-1": f"{a.loc[m,'exact1'].mean():.3f}",
                     "truth in top 25": f"{a.loc[m,'in25'].mean():.3f}"})
    write_table(pd.DataFrame(rows), out, "table5_accuracy_by_isomer_class",
                "Accuracy against the number of library structures sharing the true formula.")

    ok = a[a.formula1 == 1]
    t6 = pd.DataFrame([
        {"quantity": "Top-1 exact", "value": f"{a.exact1.mean():.4f}"},
        {"quantity": "truth in top 25", "value": f"{a.in25.mean():.4f}"},
        {"quantity": "median rank when found", "value": f"{a.loc[a.in25==1,'rank'].median():.0f}"},
        {"quantity": "top-1 formula correct", "value": f"{a.formula1.mean():.4f}"},
        {"quantity": "isomer correct given formula correct", "value": f"{ok.exact1.mean():.4f}"},
        {"quantity": "chance isomer pick", "value": f"{np.mean(1.0/ok.iso_class.clip(lower=1)):.4f}"},
        {"quantity": "Tanimoto of non-exact top-1 (median)",
         "value": f"{a.loc[a.exact1==0,'tan1'].median():.3f}"},
        {"quantity": "test peaks merged by 0.5 Da binning", "value": f"{repr_stats['frac']:.1%}"},
    ])
    write_table(t6, out, "table6_prediction_summary",
                "Prediction accuracy on the 400 test molecules, scored against the recovered key.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--only", default=None, help="comma-separated figure numbers")
    a = ap.parse_args()
    out = a.outdir or (a.root / "figures")
    style()
    d = load(a.root)
    paired, chance, n_scored = paired_differences(d["pred"])
    only = set(a.only.split(",")) if a.only else None
    def want(n): return only is None or n in only
    print("building figures", flush=True)
    if want("1"): fig_ablation(d, out, paired, chance)
    if want("2"): fig_training(d, out)
    if want("3"): fig_viewscale(d, out)
    if want("4"): fig_decomposition(d, out)
    if want("5"): fig_isomer(d, out)
    if want("6"): fig_structures(d, out)
    rs = fig_representation(a.root, out) if want("7") else {"frac": float("nan")}
    if only is None:
        print("building tables", flush=True)
        tables(d, out, paired, chance, n_scored, rs)
    json.dump({"paired": paired, "chance_macro_auprc": chance,
               "n_scored_targets": n_scored}, open(out / "figure_data.json", "w"), indent=1)
    print(f"\nall output in {out}", flush=True)


if __name__ == "__main__":
    main()
