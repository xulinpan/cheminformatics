import json, pathlib, importlib.util
import pandas as pd
spec = importlib.util.spec_from_file_location("figs", "dbf2/figures.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
OP = json.load(open("dbf2_runs_open/seed_analysis_open.json"))
SL = json.load(open("dbf2_runs/seed_sweep_analysis.json"))
RH = json.load(open("dbf2_runs/representation_heldout.json"))
RP = json.load(open("dbf2_runs_open/representation_open.json"))
out = pathlib.Path("figures")

DESC = {"M0": ("0.5 Da bins", "MLP", "mean", "1,164,560"),
        "M1R": ("0.5 Da bins", "Transformer", "mean", "712,336"),
        "M1": ("full precision", "Transformer", "mean", "712,336"),
        "M2": ("full precision", "Transformer", "hierarchical", "829,584")}
t1 = pd.DataFrame([{
    "arm": r, "mass axis": DESC[r][0], "encoder": DESC[r][1], "pooling": DESC[r][2],
    "parameters": DESC[r][3],
    "macro AUPRC": f"{OP['by_rung'][r]['mean']:.4f}",
    "SD over seeds": f"{OP['by_rung'][r]['sd']:.4f}",
} for r in ["M0", "M1R", "M1", "M2"]])
F.write_table(t1, out, "table1_arms_open",
              f"The four arms on the open corpus, each differing from the one above it "
              f"in a single respect. Macro AUPRC on {OP['n_molecules']:,} held-out molecules, "
              f"mean over three seeds. Chance is {OP['chance']:.4f} over {OP['n_scored']} "
              f"scored targets.")

NAME = {"M1R-M0": "encoder", "M1-M1R": "mass axis", "M2-M1": "aggregation",
        "M1-M0": "encoder and mass axis", "M2-M0": "all three"}
t2 = pd.DataFrame([{
    "contrast": k.replace("-", " $-$ "), "isolates": NAME[k],
    "$\\Delta$ AUPRC": f"{OP['paired'][k]['mean']:+.4f}",
    "95% CI": f"[{OP['paired'][k]['lo']:+.4f}, {OP['paired'][k]['hi']:+.4f}]",
    "targets improved": f"{OP['paired'][k]['frac_improved']:.1%}",
    "SD over seeds": (f"{OP['effects'][NAME[k]]['sd']:.4f}" if NAME[k] in OP["effects"] else "--"),
} for k in ["M1R-M0", "M1-M1R", "M2-M1", "M1-M0", "M2-M0"]])
F.write_table(t2, out, "table2_effects_open",
              "Paired per-target differences on the open corpus, bootstrapped over targets "
              "after averaging each target's average precision across the three seeds. "
              "The final column is the standard deviation of the effect across seeds.")

rows = []
for n in ["encoder", "mass axis", "aggregation"]:
    rows.append({
        "effect": n,
        "single library": f"{SL['effects'][n]['mean']:+.4f}",
        "SD (single)": f"{SL['effects'][n]['sd']:.4f}",
        "open corpus": f"{OP['effects'][n]['mean']:+.4f}",
        "SD (open)": f"{OP['effects'][n]['sd']:.4f}",
    })
sl_share = SL["paired"]["M1R-M0"]["mean"] / SL["paired"]["M1-M0"]["mean"]
op_share = OP["paired"]["M1R-M0"]["mean"] / OP["paired"]["M1-M0"]["mean"]
rows.append({"effect": "encoder share of M1 $-$ M0",
             "single library": f"{sl_share:.1%}", "SD (single)": "--",
             "open corpus": f"{op_share:.1%}", "SD (open)": "--"})
rows.append({"effect": "peaks absorbed at 0.5 Da",
             "single library": f"{100*RH['absorption_by_width']['0.5']:.1f}%", "SD (single)": "--",
             "open corpus": f"{100*RP['absorption_by_width']['0.5']:.1f}%", "SD (open)": "--"})
F.write_table(pd.DataFrame(rows), out, "table3_two_corpora",
              "The decomposition on two corpora. The single-library corpus is 99% timsTOF "
              "from one in-house library; the open corpus spans 66 instrument types across "
              "GNPS, MassBank and MoNA. Absorption is measured on each corpus's own held-out "
              "spectra.")
for n in ("table1_arms_open", "table2_effects_open", "table3_two_corpora"):
    print(f"--- {n} ---"); print(open(out / f"{n}.csv").read())
