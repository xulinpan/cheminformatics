"""Prespecified M0-only hyperparameter search.

The manuscript's argument is about comparator fairness, so a comparator that is
merely untuned is a weaker instrument than the argument needs. The shared schedule
was chosen for the peak-token Transformer and inherited by the dense baseline; this
script gives the dense baseline its own.

The design is fixed in advance and is not adaptive:

  learning rate  {3e-4, 1e-3, 3e-3}
  hidden width   {256, 512, 1024}     (the binned MLP only; d_model is unchanged)
  dropout        {0.1, 0.3}

18 configurations, one seed each, fitted on folds 2-4 and scored on fold 1. The
best fold-1 macro AUPRC wins. Selection runs pass --no-test, so the held-out fold
is never evaluated during the search and cannot contaminate it. The winner is then
retrained under the three seeds used elsewhere in this work and evaluated once on
fold 0.

    python scripts/tune_m0.py --root . --dataset open            # stage 1, the grid
    python scripts/tune_m0.py --root . --dataset open --final    # stage 2, the winner

Only the hidden width varies M0's capacity: d_model stays at 128 so that the
representation handed to the aggregator is the same size as in the untuned arm, and
the tuned baseline therefore differs from the untuned one in the intended respects
only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

LRS = [3e-4, 1e-3, 3e-3]
WIDTHS = [256, 512, 1024]
DROPOUTS = [0.1, 0.3]
SEEDS = [2026, 7, 13]
RUNS = {"default": "dbf2_runs", "open": "dbf2_runs_open", "msg": "dbf2_runs_msg"}


def tag_for(lr: float, width: int, dropout: float) -> str:
    return f"M0tune_lr{lr:g}_h{width}_d{dropout:g}"


def run(root: Path, dataset: str, tag: str, extra: list[str]) -> None:
    cmd = [sys.executable, "-m", "dbf2", "train", "--root", str(root),
           "--dataset", dataset, "--rung", "M0", "--epochs", "15",
           "--tag", tag] + extra
    print("  " + " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True, cwd=str(root))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--dataset", default="open", choices=sorted(RUNS))
    ap.add_argument("--final", action="store_true",
                    help="stage 2: retrain the selected configuration under three "
                         "seeds and evaluate once on the held-out fold")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    runs = root / RUNS[a.dataset]
    sel_path = runs / "m0_tuning.json"

    if not a.final:
        grid = list(itertools.product(LRS, WIDTHS, DROPOUTS))
        print(f"stage 1: {len(grid)} configurations, validation only\n")
        rows = []
        for lr, width, dropout in grid:
            tag = tag_for(lr, width, dropout)
            rep = runs / tag / f"{tag}_report.json"
            if not rep.exists():
                run(root, a.dataset, tag,
                    ["--lr", str(lr), "--hidden", str(width),
                     "--dropout", str(dropout), "--no-test"])
            d = json.loads(rep.read_text())
            if "test" in d:
                raise RuntimeError(
                    f"{tag} evaluated the held-out fold; selection would be "
                    f"contaminated. Delete the run and repeat with --no-test.")
            rows.append({"lr": lr, "hidden": width, "dropout": dropout, "tag": tag,
                         "val_macro_auprc": d["best_val_macro_auprc"],
                         "n_parameters": d["n_parameters"]})
            print(f"    fold-1 macro AUPRC {rows[-1]['val_macro_auprc']:.4f}\n",
                  flush=True)

        rows.sort(key=lambda r: -r["val_macro_auprc"])
        print("\n  ranked by fold-1 macro AUPRC (fold 0 not consulted):")
        for r in rows:
            print(f"    {r['val_macro_auprc']:.4f}  lr={r['lr']:g} "
                  f"hidden={r['hidden']} dropout={r['dropout']:g}")
        best = rows[0]
        sel_path.write_text(json.dumps(
            {"grid": {"lr": LRS, "hidden": WIDTHS, "dropout": DROPOUTS},
             "selection_metric": "fold-1 macro AUPRC",
             "held_out_fold_consulted": False,
             "results": rows, "selected": best}, indent=2), encoding="utf-8")
        print(f"\n  selected: lr={best['lr']:g} hidden={best['hidden']} "
              f"dropout={best['dropout']:g}  ({best['val_macro_auprc']:.4f})")
        print(f"  wrote {sel_path}\n  now rerun with --final")
        return

    sel = json.loads(sel_path.read_text())["selected"]
    print(f"stage 2: lr={sel['lr']:g} hidden={sel['hidden']} "
          f"dropout={sel['dropout']:g}, three seeds, evaluated on fold 0\n")
    out = []
    for seed in SEEDS:
        tag = "M0T" if seed == SEEDS[0] else f"M0T_s{seed}"
        rep = runs / tag / f"{tag}_report.json"
        if not rep.exists():
            run(root, a.dataset, tag,
                ["--lr", str(sel["lr"]), "--hidden", str(sel["hidden"]),
                 "--dropout", str(sel["dropout"]), "--seed", str(seed)])
        d = json.loads(rep.read_text())
        out.append({"seed": seed, "tag": tag,
                    "macro_auprc": d["test"]["macro_auprc"],
                    "n_parameters": d["n_parameters"]})
        print(f"    seed {seed}: held-out macro AUPRC "
              f"{out[-1]['macro_auprc']:.4f}\n", flush=True)
    import statistics as st
    v = [r["macro_auprc"] for r in out]
    print(f"\n  tuned M0: {st.mean(v):.4f} +/- {st.stdev(v):.4f}")
    d = json.loads(sel_path.read_text())
    d["final"] = {"runs": out, "mean": st.mean(v), "sd": st.stdev(v)}
    sel_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"  wrote {sel_path}")


if __name__ == "__main__":
    main()
