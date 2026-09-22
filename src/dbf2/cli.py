"""Command-line entry point.

    python -m dbf2 prepare  --root . --budget 600      # resumable: rerun to continue
    python -m dbf2 train    --root . --rung M2 --epochs 20
    python -m dbf2 bayes    --root . --run dbf2_runs/M2
    python -m dbf2 ablate   --root . --rungs M0 M1R M1 M2 --epochs 15
    python -m dbf2 ablate   --root . --seed 7                 # replicate at another seed
    python -m dbf2 train    --root . --rung M1 --quantise 0.1 --tag W0.1   # bin-width sweep
    python -m dbf2 descriptors --root . --library processed_v2/molecules.parquet
    python -m dbf2 predict  --root . --run dbf2_runs/M2 --budget 150
    python -m dbf2 oracle   --root .
    python -m dbf2 prepare  --root . --dataset open      # redistributable subset
    python -m dbf2 ablate   --root . --dataset open --rungs M0 M1R M1 M2 --epochs 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Config


def cmd_prepare(a) -> None:
    from .prepare import run
    cfg = Config(paths=Config.paths_for(a.dataset, a.root))
    if a.dataset == "msg":
        # The oracle recovers the CASMI answer key by matching test spectra against
        # the training file. MassSpecGym has no competition test set, and its own
        # held-out fold is not a leak to be recovered, so there is nothing to build.
        cfg.data.build_oracle = False
    if a.max_peaks is not None:
        cfg.data.max_peaks = a.max_peaks
    if a.max_row_groups:
        cfg.data.limit_row_groups = a.max_row_groups
    run(cfg, a.budget)


def cmd_oracle(a) -> None:
    from .prepare import build_oracle_key
    cfg = Config(paths=Config.paths_for(a.dataset, a.root))
    key = build_oracle_key(cfg)
    print(key.head().to_string())
    print(f"\nwrote {cfg.paths.oracle}")
    print("This key exists to EXCLUDE these structures from training and to evaluate "
          "honestly. It must not be used to produce a competition submission.")


def _apply_overrides(cfg, a) -> None:
    """Command-line overrides shared by train and ablate."""
    if a.epochs: cfg.train.epochs = a.epochs
    if a.batch: cfg.train.batch_molecules = a.batch
    if a.views: cfg.train.views_per_molecule = a.views
    if getattr(a, "max_peaks", None) is not None: cfg.data.max_peaks = a.max_peaks
    if a.limit_molecules: cfg.train.limit_molecules = a.limit_molecules
    if getattr(a, "seed", None) is not None:
        cfg.train.seed = a.seed
    if getattr(a, "quantise", None) is not None:
        # sweep the mass axis: 0 restores full precision, any positive width
        # snaps peaks to that grid and merges the collisions
        cfg.model.quantise_mz = a.quantise
    # knobs for the prespecified M0-only tuning search
    if getattr(a, "lr", None) is not None: cfg.train.lr = a.lr
    if getattr(a, "dropout", None) is not None: cfg.model.dropout = a.dropout
    if getattr(a, "hidden", None) is not None: cfg.model.binned_hidden = a.hidden
    if getattr(a, "no_test", False):
        # Selection must not see the held-out fold. With this set the run reports
        # validation only, so a hyperparameter search cannot consult fold 0 even
        # by accident.
        cfg.train.skip_test = True


def cmd_train(a) -> None:
    from .train import train
    cfg = Config.ablation(a.rung, a.root, a.dataset)
    _apply_overrides(cfg, a)
    tag = a.tag or a.rung
    out = cfg.paths.runs / tag
    cfg.to_json(out / "config.json")
    train(cfg, out, tag)


def cmd_bayes(a) -> None:
    from .bayes_stage import run_stage2
    cfg = Config.ablation("M3", a.root)
    if a.iter: cfg.bayes.n_iter = a.iter
    if a.burn: cfg.bayes.n_burn = a.burn
    if a.subsample: cfg.bayes.subsample_molecules = a.subsample
    run_stage2(cfg, a.run)


def cmd_predict(a) -> None:
    from .predict import run
    cfg = Config.ablation("M2", a.root, a.dataset)
    if a.max_candidates: cfg.infer.max_candidates = a.max_candidates
    run(cfg, a.run, a.budget, a.calibration_queries)


def cmd_descriptors(a) -> None:
    from .candidate_descriptors import run
    cfg = Config(paths=Config.paths_for(a.dataset, a.root))
    run(cfg, a.library, a.budget)


def cmd_ablate(a) -> None:
    from .train import train
    results = {}
    for rung in a.rungs:
        cfg = Config.ablation(rung, a.root, a.dataset)
        _apply_overrides(cfg, a)
        tag = rung if a.seed is None else f"{rung}_s{a.seed}"
        out = cfg.paths.runs / tag
        cfg.to_json(out / "config.json")
        results[tag] = train(cfg, out, tag)
    name = "ablation_summary.json" if a.seed is None else f"ablation_summary_s{a.seed}.json"
    path = Config.paths_for(a.dataset, a.root).runs / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== ABLATION SUMMARY ===")
    for rung, r in results.items():
        t = r.get("test", {})
        print(f"{rung:8s}: encoder={r['encoder']:8s} pooling={r['pooling']:12s} "
              f"macro_auprc={t.get('macro_auprc', float('nan')):.4f} "
              f"ece={t.get('ece', float('nan')):.4f}")
    print(f"\nwrote {path}")


def main() -> None:
    # --root is accepted both before and after the subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default="default",
                        choices=sorted(Config.DATASETS),
                        help="which corpus to use; 'open' is the "
                             "GNPS+MassBank+MoNA subset, with its own "
                             "prepared/ and runs/ directories")
    common.add_argument("--root", type=Path, default=None,
                        help="project root holding data/ (default: current directory)")

    ap = argparse.ArgumentParser(prog="dbf2", description=__doc__,
                                 parents=[common],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", parents=[common]); p.set_defaults(fn=cmd_prepare)
    p.add_argument("--budget", type=float, default=float("inf"))
    p.add_argument("--max-peaks", type=int, default=None)
    p.add_argument("--max-row-groups", type=int, default=None,
                   help="use only the first N row groups; for smoke tests")

    p = sub.add_parser("oracle", parents=[common]); p.set_defaults(fn=cmd_oracle)

    p = sub.add_parser("train", parents=[common]); p.set_defaults(fn=cmd_train)
    p.add_argument("--rung", default="M2", choices=["M0", "M1", "M1R", "M1D", "M2", "M3", "M4", "M5"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--views", type=int, default=None)
    p.add_argument("--max-peaks", type=int, default=None)
    p.add_argument("--limit-molecules", type=int, default=None,
                   help="cap each split; for smoke tests")
    p.add_argument("--tag", default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="training seed; vary it to measure run-to-run spread")
    p.add_argument("--quantise", type=float, default=None,
                   help="snap peak m/z to this grid in Da and merge collisions; "
                        "0 keeps full precision. Sweep it to price precision.")
    p.add_argument("--lr", type=float, default=None,
                   help="override the peak learning rate")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None,
                   help="hidden width of the binned MLP; M0 only")
    p.add_argument("--no-test", action="store_true",
                   help="evaluate on validation only, leaving the held-out fold "
                        "untouched; use for hyperparameter selection")

    p = sub.add_parser("bayes", parents=[common]); p.set_defaults(fn=cmd_bayes)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--iter", type=int, default=None)
    p.add_argument("--burn", type=int, default=None)
    p.add_argument("--subsample", type=int, default=None)

    p = sub.add_parser("descriptors", parents=[common]); p.set_defaults(fn=cmd_descriptors)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--budget", type=float, default=float("inf"))

    p = sub.add_parser("predict", parents=[common]); p.set_defaults(fn=cmd_predict)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--budget", type=float, default=float("inf"))
    p.add_argument("--calibration-queries", type=int, default=400)
    p.add_argument("--max-candidates", type=int, default=None)

    p = sub.add_parser("ablate", parents=[common]); p.set_defaults(fn=cmd_ablate)
    p.add_argument("--rungs", nargs="+", default=["M0", "M1D", "M1R", "M1", "M2"])
    p.add_argument("--seed", type=int, default=None,
                   help="training seed; runs are tagged <rung>_s<seed>")
    p.add_argument("--quantise", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--views", type=int, default=None)
    p.add_argument("--limit-molecules", type=int, default=None,
                   help="cap each split; for smoke tests")

    a = ap.parse_args()
    if a.root is None:
        a.root = Path(".")
    a.fn(a)


if __name__ == "__main__":
    main()
