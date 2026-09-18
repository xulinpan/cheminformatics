"""Candidate ranking under the calibrated generalized posterior
(specification sections 9, 12.1).

    S_lambda(G; X) = lambda_0 log p0(G) + lambda_B s_B + lambda_C s_C
                     + lambda_F s_F + lambda_M s_M + M_valid(G)

The weights are nonnegative and fitted on held-out validation queries by
equation (37); they are never set by hand.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evaluate import bootstrap_ci, mrr_at_k


def bayes_structural_score(prob: np.ndarray, descriptors: np.ndarray,
                           eps: float = 1e-6) -> np.ndarray:
    """s_B for one query, equation (30) with a single posterior summary.

    prob        (J,)        posterior presence probability per target
    descriptors (K, J)      0/1 descriptors of each candidate
    """
    p = np.clip(prob, eps, 1.0 - eps)
    return descriptors @ np.log(p) + (1.0 - descriptors) @ np.log1p(-p)


def bayes_structural_score_mc(prob_draws: np.ndarray, descriptors: np.ndarray,
                              eps: float = 1e-6) -> np.ndarray:
    """s_B averaged over posterior draws, equation (30) proper.

    Averaging inside the logarithm over shared draws preserves the dependence the
    common factors induce, which independently marginalised probabilities discard.
    """
    s, j = prob_draws.shape
    ll = np.empty((s, descriptors.shape[0]))
    for i in range(s):
        ll[i] = bayes_structural_score(prob_draws[i], descriptors, eps)
    m = ll.max(axis=0)
    return m + np.log(np.exp(ll - m).mean(axis=0))


def informative_targets(prob: np.ndarray, prevalence: np.ndarray,
                        min_prevalence: float = 0.01) -> np.ndarray:
    """Restrict the product of equation (30) to targets carrying posterior mass.

    Specification section 5.6: the product over 600 terms is dominated by the rarest
    targets, so a prevalence floor is applied and the choice fixed on validation data.
    """
    return np.where(prevalence >= min_prevalence)[0]


class GeneralizedPosterior:
    """Weighted combination of evidence terms with nonnegative weights."""

    TERMS = ("prior", "bayes", "contrastive", "forward", "formula")

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = {k: float(weights.get(k, 0.0)) if weights else 0.0
                        for k in self.TERMS}
        if weights is None:
            self.weights["bayes"] = 1.0

    def score(self, terms: Dict[str, np.ndarray]) -> np.ndarray:
        total = None
        for k, w in self.weights.items():
            if w == 0.0 or k not in terms or terms[k] is None:
                continue
            v = np.asarray(terms[k], dtype=np.float64)
            total = v * w if total is None else total + v * w
        if total is None:
            raise ValueError("no active evidence terms")
        return total

    def rank(self, terms: Dict[str, np.ndarray], k: int = 25) -> np.ndarray:
        s = self.score(terms)
        return np.argsort(-s)[:k]

    # ------------------------------------------------------- weight calibration
    def calibrate(self, queries: List[Dict[str, np.ndarray]], truth_index: Sequence[int],
                  l2: float = 1e-3, n_steps: int = 400, lr: float = 0.05,
                  verbose: bool = False) -> Dict[str, float]:
        """Fit nonnegative lambda by projected gradient on the validation
        conditional likelihood of equation (37).

        Each query supplies the evidence terms for its candidate set and the index
        of the true structure within it. Queries whose candidate set misses the
        truth carry no gradient and are skipped, so the fitted weights describe
        ranking only; recall is reported separately.
        """
        active = [k for k in self.TERMS
                  if any(k in q and q[k] is not None for q in queries)]
        w = np.array([max(self.weights[k], 1e-3) for k in active], dtype=np.float64)
        usable = [(q, t) for q, t in zip(queries, truth_index)
                  if t is not None and t >= 0]
        if not usable:
            return dict(self.weights)

        for step in range(n_steps):
            grad = np.zeros_like(w)
            loss = 0.0
            for q, t in usable:
                M = np.stack([np.asarray(q[k], np.float64) for k in active])  # (T, K)
                s = w @ M
                s = s - s.max()
                p = np.exp(s); p /= p.sum()
                loss -= np.log(max(p[t], 1e-300))
                grad += M @ p - M[:, t]
            grad = grad / len(usable) + 2.0 * l2 * w
            w = np.maximum(w - lr * grad, 0.0)
            if verbose and (step + 1) % 100 == 0:
                print(f"[calibrate] step {step+1} loss={loss/len(usable):.4f} "
                      f"w={dict(zip(active, np.round(w, 3)))}", flush=True)
        for k in self.TERMS:
            self.weights[k] = float(w[active.index(k)]) if k in active else 0.0
        return dict(self.weights)


def evaluate_retrieval(ranked_keys: Sequence[Sequence[str]], truth_keys: Sequence[str],
                       k: int = 25) -> dict:
    """MRR@k, top-n hit rates and a molecule-level bootstrap interval."""
    out = mrr_at_k(ranked_keys, truth_keys, k)
    rr = []
    for cand, t in zip(ranked_keys, truth_keys):
        r = 0.0
        for i, c in enumerate(cand[:k], start=1):
            if c == t:
                r = 1.0 / i
                break
        rr.append(r)
    lo, hi = bootstrap_ci(rr)
    out["mrr_at_25_bootstrap_95ci"] = [lo, hi]
    # conditional on the truth being retrievable at all, which separates the two
    # distinct failure modes of section 14
    present = [i for i, (c, t) in enumerate(zip(ranked_keys, truth_keys)) if t in set(c)]
    out["n_with_truth_in_candidates"] = len(present)
    out["mrr_at_25_given_truth_present"] = (
        float(np.mean([rr[i] for i in present])) if present else float("nan"))
    return out
