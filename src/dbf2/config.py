"""Central configuration for DeepBayesFrag-MS v2.

Every tunable quantity in the specification lives here so that ablations differ
by configuration only, never by edited code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Sequence


# --------------------------------------------------------------------------- paths
@dataclass
class Paths:
    root: Path = Path(".")
    # Which corpus file to read, and where its derived artefacts live. The
    # defaults reproduce the original single-library runs; the "open" dataset
    # (GNPS + MassBank + MoNA) uses its own prepared/ and runs/ directories so
    # the two never mix and neither clobbers the other.
    train_name: str = "train.parquet"
    prepared_name: str = "dbf2_prepared"
    runs_name: str = "dbf2_runs"

    @property
    def raw(self) -> Path: return self.root / "data"
    @property
    def train_parquet(self) -> Path: return self.raw / self.train_name
    @property
    def test_parquet(self) -> Path: return self.raw / "test.parquet"
    @property
    def sample_submission(self) -> Path: return self.raw / "sample_submission.csv"
    @property
    def prepared(self) -> Path: return self.root / self.prepared_name
    @property
    def views(self) -> Path: return self.prepared / "views"
    @property
    def molecules(self) -> Path: return self.prepared / "molecules.parquet"
    @property
    def targets(self) -> Path: return self.prepared / "targets.parquet"
    @property
    def target_dict(self) -> Path: return self.prepared / "target_dictionary.csv"
    @property
    def oracle(self) -> Path: return self.prepared / "oracle_test_key.csv"
    @property
    def runs(self) -> Path: return self.root / self.runs_name


# ------------------------------------------------------------------- preprocessing
@dataclass
class DataConfig:
    """Stage 0. Spectral cleaning and view-set construction."""
    max_precursor_mz: float = 2000.0
    max_precursor_ppm: float = 20.0          # drop training rows with worse calibration
    min_peaks: int = 3
    rel_intensity_floor: float = 1e-3        # fraction of base peak
    max_peaks: int = 256                     # 0 or None disables the cap (spec default: no cap)
    precursor_window: float = 1.5            # drop peaks at/above precursor + this, in Da
    n_folds: int = 5
    seed: int = 2026
    limit_row_groups: Optional[int] = None   # smoke testing only; None means all

    # target vocabulary
    brics_top_k: int = 512
    min_fit_molecules: int = 10
    min_prevalence: float = 0.01             # keep targets above this for the Bayesian stage

    # leak-safe holdout (see spec Stage 0)
    build_oracle: bool = True
    exclude_oracle_from_training: bool = True


# ----------------------------------------------------------------------- the model
@dataclass
class ModelConfig:
    """Modules I-III."""
    # --- encoder (Module I)
    encoder: str = "peakset"                 # {"peakset", "binned"}  binned reproduces M0
    d_model: int = 128
    n_blocks: int = 4
    n_heads: int = 8
    d_ff: int = 256
    dropout: float = 0.1
    fourier_scales: int = 32
    lambda_min: float = 2e-3                 # shortest wavelength, Da (resolves mass defect)
    lambda_max: float = 1e3
    # binned-encoder settings, used only when encoder == "binned"
    bin_width: float = 0.5
    bin_max_mz: float = 1024.0
    # Rounding control (rung M1R). When > 0, every peak m/z is snapped to the
    # centre of its bin_width grid cell and peaks that then collide are merged
    # before the peak-set encoder ever sees them. The encoder, its parameter
    # count and the training budget are unchanged, so the only quantity that
    # differs from M1 is the precision of the mass axis. 0 disables it.
    quantise_mz: float = 0.0
    quantise_merge: str = "sqrt"             # {"sqrt", "sum", "max"}; sqrt matches BinnedEncoder

    # --- acquisition covariates
    d_cov: int = 32
    energy_spline_knots: int = 6
    energy_scale: float = 100.0

    # --- multi-view aggregation (Module II)
    pooling: str = "hierarchical"            # {"hierarchical", "mean"}  mean reproduces M1
    beta_floor: float = 1e-2                 # numerical floor on sensitivity
    log_sigma_clamp: float = 4.0             # |log sigma| bound
    max_precision: float = 1e3               # clamp on w = beta^2 / sigma^2
    learn_prior_scale: bool = True

    # --- targets
    n_presence: int = 512
    n_count: int = 88
    count_head: bool = True

    # --- contrastive alignment (Module VI); off by default in the minimal experiment
    contrastive: bool = False
    d_proj: int = 128
    temperature: float = 0.1


@dataclass
class TrainConfig:
    """Stage 1."""
    epochs: int = 20
    patience: int = 4
    batch_molecules: int = 16                # attention cost scales with batch x views
    views_per_molecule: int = 4              # random subsample during training
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 0
    train_folds: Sequence[int] = (2, 3, 4)
    val_fold: int = 1
    test_fold: int = 0
    contrastive_weight: float = 0.1
    count_weight: float = 0.3
    seed: int = 2026
    limit_molecules: Optional[int] = None    # cap each split; for smoke tests


@dataclass
class BayesConfig:
    """Stage 2: Polya-Gamma augmented low-rank BART residual."""
    n_factors: int = 8
    n_trees: int = 20
    alpha_tree: float = 0.95
    beta_tree: float = 2.0
    n_iter: int = 2000
    n_burn: int = 1000
    thin: int = 5
    subsample_molecules: int = 20000         # spec 21.2: subsample for feasibility
    max_targets: Optional[int] = None        # None -> all targets above min_prevalence
    leaf_prior_sd: float = 0.5
    horseshoe_global_scale: float = 0.1
    residual_sd_init: float = 0.5
    seed: int = 2026


@dataclass
class InferenceConfig:
    """Modules IV, V and the generalized posterior."""
    formula_ppm: float = 10.0
    rdbe_range: tuple = (-0.5, 40.0)
    elements: tuple = ("C", "H", "N", "O", "P", "S", "Cl", "Br", "F")
    candidate_ppm: float = 10.0
    max_candidates: int = 2000
    top_k: int = 25
    # generalized-posterior weights; fitted by calibrate_lambda, never hand-set
    lambda_init: dict = field(default_factory=lambda: {
        "prior": 0.0, "bayes": 1.0, "contrastive": 0.0, "forward": 0.0, "formula": 1.0})


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bayes: BayesConfig = field(default_factory=BayesConfig)
    infer: InferenceConfig = field(default_factory=InferenceConfig)

    def train_folds_for_vocab(self) -> list:
        """Folds the target vocabulary and every scaler may be fitted on."""
        return list(self.train.train_folds)

    def to_json(self, path: Path) -> None:
        d = asdict(self)
        d["paths"] = {"root": str(self.paths.root)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")

    #: Named corpora. "open" is the redistributable subset built by
    #: scripts/make_open_subset.py; see docs/corpus_composition.md.
    DATASETS = {
        "default": ("train.parquet", "dbf2_prepared", "dbf2_runs"),
        "open": ("train_open.parquet", "dbf2_prepared_open", "dbf2_runs_open"),
    }

    @staticmethod
    def paths_for(dataset: str, root) -> "Paths":
        if dataset not in Config.DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; "
                             f"expected one of {sorted(Config.DATASETS)}")
        train, prep, runs = Config.DATASETS[dataset]
        return Paths(root=Path(root), train_name=train,
                     prepared_name=prep, runs_name=runs)

    @staticmethod
    def ablation(rung: str, root, dataset: str = "default") -> "Config":
        """Configurations for the ablation ladder of specification section 15."""
        c = Config(paths=Config.paths_for(dataset, root))
        if rung == "M0":        # binned vector, mean-pooled views
            c.model.encoder, c.model.pooling = "binned", "mean"
        elif rung == "M1":      # peak-set encoder, still mean-pooled
            c.model.encoder, c.model.pooling = "peakset", "mean"
        elif rung == "M1R":     # M1 with the mass axis rounded to the M0 bin grid
            c.model.encoder, c.model.pooling = "peakset", "mean"
            c.model.quantise_mz = c.model.bin_width
        elif rung in ("M2", "M3", "M4", "M5"):
            c.model.encoder, c.model.pooling = "peakset", "hierarchical"
        else:
            raise ValueError(f"unknown ablation rung {rung!r}; expected M0, M1, M1R, M2..M5")
        return c
