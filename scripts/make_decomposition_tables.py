import json, pathlib, importlib.util
import pandas as pd
spec = importlib.util.spec_from_file_location("figs", "dbf2/figures.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
A = json.load(open("dbf2_runs/seed_sweep_analysis.json")); out = pathlib.Path("figures")

DESC = {"M0":  ("0.5 Da bins", "MLP", "mean", "1,164,560"),
        "M1R": ("0.5 Da bins", "Transformer", "mean", "712,336"),
        "M1":  ("full precision", "Transformer", "mean", "712,336"),
        "M2":  ("full precision", "Transformer", "hierarchical", "829,584")}
t1 = pd.DataFrame([{
    "arm": r, "mass axis": DESC[r][0], "encoder": DESC[r][1], "pooling": DESC[r][2],
    "parameters": DESC[r][3],
    "macro AUPRC": f"{A['by_rung'][r]['mean']:.4f}",
    "SD over seeds": f"{A['by_rung'][r]['sd']:.4f}",
} for r in ["M0", "M1R", "M1", "M2"]])
F.write_table(t1, out, "table1_arms",
              f"The four arms, each differing from the one above it in a single "
              f"respect. Macro AUPRC on {8692:,} held-out molecules, mean over three "
              f"training seeds. Chance is {A['chance']:.4f} over {A['n_scored']} scored targets.")

NAME = {"M1R-M0": "encoder", "M1-M1R": "mass axis", "M2-M1": "aggregation",
        "M1-M0": "encoder and mass axis", "M2-M0": "all three"}
t2 = pd.DataFrame([{
    "contrast": k.replace("-", " $-$ "), "isolates": NAME[k],
    "$\\Delta$ AUPRC": f"{A['paired'][k]['mean']:+.4f}",
    "95\\% CI": f"[{A['paired'][k]['lo']:+.4f}, {A['paired'][k]['hi']:+.4f}]",
    "targets improved": f"{A['paired'][k]['frac_improved']:.1%}",
    "SD over seeds": (f"{A['effects'][NAME[k]]['sd']:.4f}" if NAME[k] in A["effects"] else "--"),
} for k in ["M1R-M0", "M1-M1R", "M2-M1", "M1-M0", "M2-M0"]])
F.write_table(t2, out, "table2_effects",
              "Paired per-target differences in average precision, bootstrapped over "
              "targets after averaging each target's AP across the three seeds. The "
              "final column is the standard deviation of the effect across seeds.")

rows3 = []
for sw in A["sweep"]:
    w = sw["width_da"]
    key = f"{w:g}"
    rows3.append({
        "bin width (Da)": "full precision" if w == 0 else key,
        "peaks absorbed": f"{100*A['absorption_by_width'][key]:.1f}%",
        "macro AUPRC": f"{sw['macro_auprc']:.4f}",
        "share of the 0.5 Da gap closed": f"{sw['frac_of_precision_gain']:.0%}",
    })
t3 = pd.DataFrame(rows3)
F.write_table(t3, out, "table3_binwidth",
              "Bin-width sweep, single seed. Every arm uses the M1 encoder at an "
              "identical parameter count; only the width of the mass grid changes. "
              "The final column expresses each result as a share of the distance "
              "between the 0.5 Da arm and full precision.")
for n in ("table1_arms", "table2_effects", "table3_binwidth"):
    print(f"--- {n} ---"); print(open(out/f"{n}.csv").read())
