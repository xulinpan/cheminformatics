"""Modules IV and V: molecular formula inference and candidate-set construction
(specification sections 6 and 7).

No candidate set ships with the competition. Constructing it is part of the task,
and its recall upper-bounds every retrieval metric:

    MRR@25  <=  Pr( G_true in C(X) ).

Recall is therefore reported as a first-class number, never assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config, InferenceConfig
from . import chem


@dataclass
class StructureDB:
    """Searchable library of known structures with precomputed descriptors."""
    inchikey14: np.ndarray
    smiles: np.ndarray
    formula: np.ndarray
    mono_mass: np.ndarray
    row_of_key: Dict[str, int]
    _order: np.ndarray
    _sorted_mass: np.ndarray
    _formula_index: Dict[str, np.ndarray]

    @staticmethod
    def from_molecules(path: Path, exclude_keys: Optional[Sequence[str]] = None
                       ) -> "StructureDB":
        mol = pd.read_parquet(path, columns=["inchikey14", "normalized_smiles",
                                             "molecular_formula", "mono_mass"])
        mol = mol.dropna(subset=["mono_mass"])
        if exclude_keys:
            mol = mol[~mol.inchikey14.isin(set(exclude_keys))]
        mol = mol.reset_index(drop=True)
        mass = mol.mono_mass.to_numpy(np.float64)
        order = np.argsort(mass)
        fidx: Dict[str, List[int]] = {}
        for i, f in enumerate(mol.molecular_formula.to_numpy()):
            fidx.setdefault(f, []).append(i)
        return StructureDB(
            inchikey14=mol.inchikey14.to_numpy(),
            smiles=mol.normalized_smiles.to_numpy(),
            formula=mol.molecular_formula.to_numpy(),
            mono_mass=mass,
            row_of_key={k: i for i, k in enumerate(mol.inchikey14.to_numpy())},
            _order=order, _sorted_mass=mass[order],
            _formula_index={k: np.asarray(v, dtype=np.int64) for k, v in fidx.items()})

    def by_mass(self, neutral_mass: float, ppm: float) -> np.ndarray:
        tol = neutral_mass * ppm * 1e-6
        lo, hi = np.searchsorted(self._sorted_mass, [neutral_mass - tol, neutral_mass + tol])
        return self._order[lo:hi]

    def by_formula(self, formula: str) -> np.ndarray:
        return self._formula_index.get(formula, np.empty(0, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.inchikey14)


# ------------------------------------------------------------------ Module IV
def score_formulas(precursor_mz: float, adduct: str, peak_mz: np.ndarray,
                   peak_intensity: np.ndarray, cfg: InferenceConfig,
                   max_candidates: int = 400) -> pd.DataFrame:
    """Rank the formulas consistent with the precursor, equation (34).

    Each candidate formula is scored by the fraction of fragment intensity
    assignable to one of its subformulas, plus a mass-accuracy term. This is the
    discriminative reranking of specification section 6 with the learned term
    replaced by its deterministic features; a trained scorer can be substituted by
    supplying ``learned_score``.
    """
    cands = chem.enumerate_formulas(precursor_mz, adduct, ppm=cfg.formula_ppm,
                                    elements=cfg.elements, rdbe_range=cfg.rdbe_range,
                                    max_results=max(max_candidates, 2000))
    if not cands:
        return pd.DataFrame(columns=["formula", "neutral_mass", "ppm", "coverage", "score"])
    # Truncating by |ppm| before scoring discards the true formula in a quarter of
    # queries, because mass error alone does not order candidates usefully. Score
    # the whole enumeration and truncate afterwards.
    cands = cands[:max_candidates]
    rows = []
    add = chem.get_adduct(adduct)
    charge = add.charge if add else 1
    for f, mass, ppm_err in cands:
        cov = chem.subformula_coverage(peak_mz, peak_intensity,
                                       chem.parse_formula(f), ppm=cfg.formula_ppm,
                                       charge=charge)
        cov = 0.0 if not np.isfinite(cov) else cov
        rows.append((f, mass, ppm_err, cov))
    df = pd.DataFrame(rows, columns=["formula", "neutral_mass", "ppm", "coverage"])
    df["score"] = 3.0 * df.coverage - np.abs(df.ppm) / max(cfg.formula_ppm, 1e-6)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    e = np.exp(df.score - df.score.max())
    df["prob"] = e / e.sum()
    return df


# ------------------------------------------------------------------- Module V
def build_candidates(db: StructureDB, precursor_mz: float, adduct: str,
                     cfg: InferenceConfig,
                     formula_probs: Optional[pd.DataFrame] = None,
                     union_with_mass: bool = True,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (candidate row indices into db, log formula prior per candidate).

    Source C1 of equation (40): structures from the local library that are
    compatible with the precursor. When formula probabilities are supplied the
    formula constraint replaces the looser mass window and each candidate inherits
    its formula's log probability as the s_M term.
    """
    add = chem.get_adduct(adduct)
    if add is None or not np.isfinite(precursor_mz):
        return np.empty(0, dtype=np.int64), np.empty(0)

    if formula_probs is not None and len(formula_probs):
        idx_parts, logp_parts = [], []
        for f, p in zip(formula_probs.formula, formula_probs.prob):
            rows = db.by_formula(f)
            if rows.size:
                idx_parts.append(rows)
                logp_parts.append(np.full(rows.size, np.log(max(float(p), 1e-12))))
        if idx_parts:
            idx = np.concatenate(idx_parts)
            logp = np.concatenate(logp_parts)
            if union_with_mass:
                # Never let an uncertain formula prediction reduce recall below the
                # mass-only pool: the formula term downweights, it does not exclude.
                neutral = add.neutral_mass(float(precursor_mz))
                extra = db.by_mass(neutral, cfg.candidate_ppm)
                extra = np.setdiff1d(extra, idx, assume_unique=False)
                if extra.size:
                    idx = np.concatenate([idx, extra])
                    logp = np.concatenate([logp, np.full(extra.size, np.log(1e-4))])
            uniq, first = np.unique(idx, return_index=True)
            idx, logp = idx[np.sort(first)], logp[np.sort(first)]
            if idx.size > cfg.max_candidates:
                keep = np.argsort(-logp)[: cfg.max_candidates]
                idx, logp = idx[keep], logp[keep]
            return idx, logp

    neutral = add.neutral_mass(float(precursor_mz))
    idx = db.by_mass(neutral, cfg.candidate_ppm)
    if idx.size > cfg.max_candidates:
        err = np.abs(db.mono_mass[idx] - neutral)
        idx = idx[np.argsort(err)[: cfg.max_candidates]]
    return idx, np.zeros(idx.size)


def candidate_recall(candidate_keys: Sequence[Sequence[str]],
                     truth_keys: Sequence[str]) -> Dict[str, float]:
    """Pr(true structure in C) and the pool-size distribution, equation (39)."""
    hit = [t in set(c) for c, t in zip(candidate_keys, truth_keys)]
    sizes = np.array([len(c) for c in candidate_keys], dtype=float)
    return {"recall": float(np.mean(hit)) if hit else float("nan"),
            "n_queries": len(hit),
            "median_pool": float(np.median(sizes)) if sizes.size else float("nan"),
            "p90_pool": float(np.percentile(sizes, 90)) if sizes.size else float("nan"),
            "empty_pools": int((sizes == 0).sum())}
