"""Evaluation (specification section 14).

The decisive measurement is calibration stratified by how the molecule was
measured: the hierarchical model should show its largest gain over mean pooling
for molecules with few or homogeneous views.

Metrics are implemented directly so that scikit-learn is not a hard dependency.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# ------------------------------------------------------------------ base metrics
def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the precision-recall curve, the step-function estimator."""
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    return float((precision * y).sum() / n_pos)


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    n_pos = int(y.sum()); n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    ranks = np.empty(y.size, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1)
    # average ranks over ties
    sp = p[order]
    i = 0
    while i < sp.size:
        j = i
        while j + 1 < sp.size and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width binning ECE over the pooled (molecule, target) predictions."""
    y = y.ravel().astype(np.float64); p = p.ravel().astype(np.float64)
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 15):
    y = y.ravel(); p = p.ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        conf.append(float(p[m].mean()) if m.any() else np.nan)
        acc.append(float(y[m].mean()) if m.any() else np.nan)
        cnt.append(int(m.sum()))
    return {"confidence": conf, "observed": acc, "count": cnt}


# ------------------------------------------------------------ aggregate reporting
def presence_metrics(prob: np.ndarray, target: np.ndarray,
                     min_positives: int = 5) -> Dict[str, float]:
    """Macro metrics over targets with enough positives, plus pooled micro metrics.

    AUPRC leads because 475 of 600 targets have prevalence below one percent, where
    AUROC is uninformative on its own.
    """
    if target is None:
        return {}
    ap, auc = [], []
    for j in range(target.shape[1]):
        y = target[:, j].astype(np.float64)
        if y.sum() < min_positives or y.sum() > y.size - min_positives:
            continue
        ap.append(average_precision(y, prob[:, j]))
        auc.append(roc_auc(y, prob[:, j]))
    flat_y, flat_p = target.ravel().astype(np.float64), prob.ravel()
    return {
        "macro_auprc": float(np.nanmean(ap)) if ap else float("nan"),
        "macro_auroc": float(np.nanmean(auc)) if auc else float("nan"),
        "n_targets_scored": len(ap),
        "micro_auprc": average_precision(flat_y, flat_p),
        "brier": brier(flat_y, flat_p),
        "ece": expected_calibration_error(flat_y, flat_p),
    }


def stratified_calibration(prob: np.ndarray, target: np.ndarray,
                           strata: Dict[str, np.ndarray],
                           bins: Sequence = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 9)),
                           ) -> Dict[str, dict]:
    """Calibration within strata of a molecule-level integer covariate.

    This is the test of specification section 4.3: uncertainty should be honest
    when little evidence is available, so the sparse strata are where the
    hierarchical model must win.
    """
    out: Dict[str, dict] = {}
    for name, values in strata.items():
        rows = {}
        for lo, hi in bins:
            m = (values >= lo) & (values <= hi)
            if m.sum() < 5:
                continue
            key = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+")
            y, p = target[m], prob[m]
            rows[key] = {
                "n_molecules": int(m.sum()),
                "ece": expected_calibration_error(y.ravel().astype(float), p.ravel()),
                "brier": brier(y.ravel().astype(float), p.ravel()),
                "micro_auprc": average_precision(y.ravel().astype(float), p.ravel()),
                "mean_predicted": float(p.mean()),
                "mean_observed": float(y.mean()),
            }
        out[name] = rows
    return out


def full_report(pred: Dict[str, np.ndarray]) -> dict:
    """Aggregate plus stratified metrics for one split."""
    prob, target = pred["prob"], pred.get("target")
    if target is None:
        return {"n_molecules": int(prob.shape[0])}
    rep: dict = {"n_molecules": int(prob.shape[0]), **presence_metrics(prob, target)}
    strata = {}
    if "n_views" in pred:
        strata["views_used"] = pred["n_views"]
    if "n_views_total" in pred:
        strata["views_available"] = pred["n_views_total"]
    if "ce_distinct" in pred:
        strata["distinct_collision_energies"] = pred["ce_distinct"]
    if strata:
        rep["stratified_calibration"] = stratified_calibration(prob, target, strata)
    if "theta_var" in pred:
        v = pred["theta_var"]
        rep["posterior_variance"] = {
            "mean": float(np.mean(v)), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    rep["reliability"] = reliability_curve(target.ravel().astype(float), prob.ravel())
    return rep


# --------------------------------------------------------------- retrieval metrics
def mrr_at_k(ranked_ids: Sequence[Sequence[str]], truth: Sequence[str], k: int = 25):
    """Mean reciprocal rank and top-n hit rates over a list of ranked candidate ids."""
    rr, hits = [], {1: 0, 5: 0, 10: 0, 25: 0}
    for cand, t in zip(ranked_ids, truth):
        r = 0.0
        for i, c in enumerate(cand[:k], start=1):
            if c == t:
                r = 1.0 / i
                for n in hits:
                    if i <= n:
                        hits[n] += 1
                break
        rr.append(r)
    n = max(len(rr), 1)
    out = {"mrr_at_25": float(np.mean(rr)), "n_queries": len(rr)}
    out.update({f"top_{n_}": hits[n_] / n for n_ in hits})
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 2026,
                 alpha: float = 0.05):
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    draws = v[rng.randint(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)))
