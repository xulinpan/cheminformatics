"""Stage 2: Polya-Gamma augmented low-rank BART residual (specification section 5).

Model, conditional on the frozen encoder:

    theta_ij ~ N( mu_ij , tau_j^2 ),   mu_ij = a_ij + sum_r lambda_jr B_r(H_i)
    D_ij     ~ Bernoulli( sigmoid(theta_ij) )
    view evidence contributes the Gaussian sufficient statistics (s1_ij, s2_ij)

With omega_ij ~ PG(1, theta_ij) the full conditional for theta is Gaussian,

    precision = omega_ij + tau_j^-2 + s2_ij
    mean      = precision^-1 ( D_ij - 1/2 + mu_ij tau_j^-2 + s1_ij )

so every level of the hierarchy is conjugate and the sum-of-trees component is
fitted by ordinary Bayesian backfitting on the latent scale.

Backfitting for factor r reduces to a univariate BART problem: given the loadings,
each molecule contributes a pseudo-response with a known precision,

    P_i = sum_j lambda_jr^2 / tau_j^2 ,
    M_i = P_i^-1 sum_j lambda_jr ( R_ij - sum_{r'!=r} lambda_jr' B_r'(H_i) ) / tau_j^2 .
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:                                  # exact and fast when available
    from polyagamma import random_polyagamma as _rpg
    HAVE_POLYAGAMMA = True
except Exception:                     # pragma: no cover - optional dependency
    _rpg = None
    HAVE_POLYAGAMMA = False


# --------------------------------------------------------------- Polya-Gamma draws
def sample_pg(c: np.ndarray, rng: np.random.RandomState, n_terms: int = 20) -> np.ndarray:
    """Draw omega ~ PG(1, c) elementwise.

    Uses the exact sampler from ``polyagamma`` when installed; otherwise the
    truncated infinite-sum representation

        omega = (2 pi^2)^-1 sum_{k=1..K} g_k / ((k - 1/2)^2 + c^2 / (4 pi^2)),
        g_k ~ Exp(1),

    whose truncation bias falls as 1/K. Install ``polyagamma`` for production runs.
    """
    if HAVE_POLYAGAMMA:
        return _rpg(1.0, c, random_state=rng)
    shape = c.shape
    k = np.arange(1, n_terms + 1, dtype=np.float64)
    denom = (k - 0.5) ** 2 + (c.reshape(-1, 1) ** 2) / (4.0 * math.pi ** 2)
    g = rng.exponential(size=(c.size, n_terms))
    return ((g / denom).sum(axis=1) / (2.0 * math.pi ** 2)).reshape(shape)


# ----------------------------------------------------------------------- one tree
class Tree:
    """Binary regression tree over a fixed design matrix, with cached leaf membership."""

    __slots__ = ("feature", "threshold", "left", "right", "depth", "value",
                 "is_leaf", "alive", "members", "n_nodes")

    def __init__(self, n_obs: int):
        self.feature = [-1]
        self.threshold = [0.0]
        self.left = [-1]
        self.right = [-1]
        self.depth = [0]
        self.value = [0.0]
        self.is_leaf = [True]
        self.alive = [True]
        self.members: Dict[int, np.ndarray] = {0: np.arange(n_obs)}
        self.n_nodes = 1

    def leaves(self) -> List[int]:
        return [i for i in range(self.n_nodes) if self.alive[i] and self.is_leaf[i]]

    def prunable(self) -> List[int]:
        """Internal nodes both of whose children are live leaves."""
        out = []
        for i in range(self.n_nodes):
            if self.alive[i] and not self.is_leaf[i]:
                l, r = self.left[i], self.right[i]
                if (l >= 0 and r >= 0 and self.alive[l] and self.alive[r]
                        and self.is_leaf[l] and self.is_leaf[r]):
                    out.append(i)
        return out

    def predict(self) -> np.ndarray:
        n = sum(len(v) for v in self.members.values())
        out = np.zeros(n)
        for leaf, idx in self.members.items():
            out[idx] = self.value[leaf]
        return out


def _node_loglik(sum_py: float, sum_p: float, leaf_var: float) -> float:
    a = 1.0 / leaf_var + sum_p
    return -0.5 * math.log(leaf_var * a) + 0.5 * sum_py * sum_py / a


def _p_split(depth: int, alpha: float, beta: float) -> float:
    return min(max(alpha * (1.0 + depth) ** (-beta), 1e-12), 1 - 1e-12)


class SumOfTrees:
    """A BART ensemble fitted to a pseudo-response with known per-observation precision."""

    def __init__(self, n_obs: int, n_trees: int, alpha: float, beta: float,
                 leaf_sd: float, rng: np.random.RandomState):
        self.trees = [Tree(n_obs) for _ in range(n_trees)]
        self.alpha, self.beta = alpha, beta
        self.leaf_var = (leaf_sd / math.sqrt(max(n_trees, 1))) ** 2
        self.rng = rng
        self.fitted = np.zeros((n_trees, n_obs))
        self.accept = 0
        self.propose = 0

    def value(self) -> np.ndarray:
        return self.fitted.sum(axis=0)

    def step(self, X: np.ndarray, y: np.ndarray, prec: np.ndarray) -> None:
        """One Bayesian-backfitting sweep over the ensemble."""
        total = self.value()
        for t, tree in enumerate(self.trees):
            partial = y - (total - self.fitted[t])
            self._update_tree(tree, X, partial, prec)
            new = tree.predict()
            total = total - self.fitted[t] + new
            self.fitted[t] = new

    # ------------------------------------------------------------------ MH moves
    def _update_tree(self, tree: Tree, X: np.ndarray, y: np.ndarray,
                     prec: np.ndarray) -> None:
        leaves = tree.leaves()
        prunable = tree.prunable()
        grow = (len(prunable) == 0) or (self.rng.rand() < 0.5)
        self.propose += 1
        if grow:
            self._try_grow(tree, X, y, prec, leaves, len(prunable))
        else:
            self._try_prune(tree, y, prec, prunable, len(leaves))
        self._draw_leaves(tree, y, prec)

    def _try_grow(self, tree: Tree, X, y, prec, leaves, n_prunable_before) -> None:
        leaf = leaves[self.rng.randint(len(leaves))]
        idx = tree.members[leaf]
        if idx.size < 5:
            return
        var = self.rng.randint(X.shape[1])
        col = X[idx, var]
        lo, hi = col.min(), col.max()
        if not np.isfinite(lo) or hi <= lo:
            return
        thr = float(col[self.rng.randint(idx.size)])
        left_mask = col <= thr
        n_l = int(left_mask.sum())
        if n_l == 0 or n_l == idx.size:
            return
        il, ir = idx[left_mask], idx[~left_mask]

        py, p = y * prec, prec
        ll_parent = _node_loglik(py[idx].sum(), p[idx].sum(), self.leaf_var)
        ll_new = (_node_loglik(py[il].sum(), p[il].sum(), self.leaf_var)
                  + _node_loglik(py[ir].sum(), p[ir].sum(), self.leaf_var))

        d = tree.depth[leaf]
        ps, psc = _p_split(d, self.alpha, self.beta), _p_split(d + 1, self.alpha, self.beta)
        log_prior = math.log(ps) + 2.0 * math.log1p(-psc) - math.log1p(-ps)
        log_trans = math.log(len(leaves)) - math.log(max(n_prunable_before + 1, 1))

        if math.log(self.rng.rand() + 1e-300) < ll_new - ll_parent + log_prior + log_trans:
            l_id, r_id = tree.n_nodes, tree.n_nodes + 1
            tree.feature[leaf], tree.threshold[leaf] = var, thr
            tree.is_leaf[leaf] = False
            tree.left[leaf], tree.right[leaf] = l_id, r_id
            for nid, members in ((l_id, il), (r_id, ir)):
                tree.feature.append(-1); tree.threshold.append(0.0)
                tree.left.append(-1); tree.right.append(-1)
                tree.depth.append(d + 1); tree.value.append(0.0)
                tree.is_leaf.append(True); tree.alive.append(True)
                tree.members[nid] = members
            tree.members.pop(leaf, None)
            tree.n_nodes += 2
            self.accept += 1

    def _try_prune(self, tree: Tree, y, prec, prunable, n_leaves_before) -> None:
        node = prunable[self.rng.randint(len(prunable))]
        l_id, r_id = tree.left[node], tree.right[node]
        il, ir = tree.members[l_id], tree.members[r_id]
        idx = np.concatenate([il, ir])
        py, p = y * prec, prec
        ll_merged = _node_loglik(py[idx].sum(), p[idx].sum(), self.leaf_var)
        ll_split = (_node_loglik(py[il].sum(), p[il].sum(), self.leaf_var)
                    + _node_loglik(py[ir].sum(), p[ir].sum(), self.leaf_var))
        d = tree.depth[node]
        ps, psc = _p_split(d, self.alpha, self.beta), _p_split(d + 1, self.alpha, self.beta)
        log_prior = math.log1p(-ps) - math.log(ps) - 2.0 * math.log1p(-psc)
        log_trans = math.log(len(prunable)) - math.log(max(n_leaves_before - 1, 1))
        if math.log(self.rng.rand() + 1e-300) < ll_merged - ll_split + log_prior + log_trans:
            tree.is_leaf[node] = True
            tree.feature[node] = -1
            tree.left[node] = tree.right[node] = -1
            tree.alive[l_id] = tree.alive[r_id] = False
            tree.members[node] = idx
            tree.members.pop(l_id, None); tree.members.pop(r_id, None)
            self.accept += 1

    def _draw_leaves(self, tree: Tree, y, prec) -> None:
        for leaf, idx in tree.members.items():
            sp = prec[idx].sum()
            spy = (prec[idx] * y[idx]).sum()
            a = 1.0 / self.leaf_var + sp
            tree.value[leaf] = float(spy / a + self.rng.randn() / math.sqrt(a))


# ------------------------------------------------------------------- full sampler
@dataclass
class PGBARTState:
    theta: np.ndarray
    loadings: np.ndarray       # (J, R)
    factors: np.ndarray        # (n, R)
    tau: np.ndarray            # (J,)
    xi: np.ndarray             # horseshoe local scales (J, R)
    tau_global: float


class PGBARTSampler:
    """Polya-Gamma Gibbs sampler for the low-rank BART residual model."""

    def __init__(self, H: np.ndarray, D: np.ndarray, offset: np.ndarray,
                 s1: Optional[np.ndarray], s2: Optional[np.ndarray],
                 n_factors: int = 8, n_trees: int = 20, alpha: float = 0.95,
                 beta: float = 2.0, leaf_sd: float = 0.5,
                 global_scale: float = 0.1, seed: int = 2026,
                 tau_prior_shape: float = 3.0, tau_prior_rate: float = 1.0,
                 tau_bounds: Tuple[float, float] = (0.05, 3.0)):
        self.H = np.ascontiguousarray(H, dtype=np.float64)
        self.D = np.ascontiguousarray(D, dtype=np.float64)
        self.offset = np.ascontiguousarray(offset, dtype=np.float64)
        self.n, self.J = self.D.shape
        self.R = n_factors
        self.s1 = np.zeros_like(self.D) if s1 is None else np.asarray(s1, np.float64)
        self.s2 = np.zeros_like(self.D) if s2 is None else np.asarray(s2, np.float64)
        self.rng = np.random.RandomState(seed)
        self.global_scale = global_scale
        self.tau_a0, self.tau_b0 = tau_prior_shape, tau_prior_rate
        self.tau_lo, self.tau_hi = tau_bounds

        self.theta = self.offset.copy()
        self.loadings = self.rng.randn(self.J, self.R) * 0.05
        self.tau = np.ones(self.J)
        self.xi = np.ones((self.J, self.R))
        self.tau_global = global_scale
        self.ensembles = [SumOfTrees(self.n, n_trees, alpha, beta, leaf_sd, self.rng)
                          for _ in range(self.R)]
        self.factors = np.zeros((self.n, self.R))

    # ------------------------------------------------------------------- one sweep
    def step(self) -> None:
        kappa = self.D - 0.5
        omega = sample_pg(self.theta, self.rng)
        omega = np.clip(omega, 1e-8, 1e8)

        mu = self.offset + self.factors @ self.loadings.T
        tau2_inv = 1.0 / np.maximum(self.tau ** 2, 1e-8)

        prec = omega + tau2_inv[None, :] + self.s2
        mean = (kappa + mu * tau2_inv[None, :] + self.s1) / prec
        self.theta = mean + self.rng.randn(self.n, self.J) / np.sqrt(prec)

        resid = self.theta - self.offset
        for r in range(self.R):
            lam = self.loadings[:, r]
            others = resid - (self.factors @ self.loadings.T
                              - np.outer(self.factors[:, r], lam))
            w = lam ** 2 * tau2_inv
            P = w.sum()
            if P < 1e-10:
                self.factors[:, r] = 0.0
                continue
            M = (others * (lam * tau2_inv)[None, :]).sum(axis=1) / P
            self.ensembles[r].step(self.H, M, np.full(self.n, P))
            self.factors[:, r] = self.ensembles[r].value()

        self._draw_loadings(resid, tau2_inv)
        self._draw_tau(resid)
        self._draw_horseshoe()

    def _draw_loadings(self, resid: np.ndarray, tau2_inv: np.ndarray) -> None:
        B = self.factors
        BtB = B.T @ B
        BtR = B.T @ resid                                   # (R, J)
        for j in range(self.J):
            prior_prec = np.diag(1.0 / np.maximum(
                (self.tau_global ** 2) * self.xi[j] ** 2, 1e-10))
            A = BtB * tau2_inv[j] + prior_prec
            try:
                L = np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                A = A + np.eye(self.R) * 1e-6
                L = np.linalg.cholesky(A)
            b = BtR[:, j] * tau2_inv[j]
            m = np.linalg.solve(L.T, np.linalg.solve(L, b))
            self.loadings[j] = m + np.linalg.solve(L.T, self.rng.randn(self.R))

    def _draw_tau(self, resid: np.ndarray) -> None:
        """Inverse-gamma draw for the latent residual scale, then a hard bound.

        tau_j is the spread of the latent log-odds not explained by the factors.
        With a single Bernoulli observation per cell it is only weakly identified:
        without the view-evidence term (s2 > 0) the sampler can drive tau upward,
        detach theta from mu, and separate. The weakly informative prior below plus
        the stated bound prevent that. The bound is on the log-odds scale, where a
        residual standard deviation above about three carries no scientific meaning.
        """
        r = resid - self.factors @ self.loadings.T
        a = self.tau_a0 + 0.5 * self.n
        b = self.tau_b0 + 0.5 * (r ** 2).sum(axis=0)
        tau2 = b / self.rng.gamma(a, size=self.J)
        self.tau = np.clip(np.sqrt(np.maximum(tau2, 1e-8)), self.tau_lo, self.tau_hi)

    def _draw_horseshoe(self) -> None:
        """Makalic and Schmidt auxiliary-variable representation: all conditionals
        are inverse gamma, so the half-Cauchy prior is sampled conjugately."""
        lam2 = self.loadings ** 2
        nu = 1.0 / self.rng.gamma(1.0, 1.0 / (1.0 + 1.0 / self.xi ** 2))
        self.xi = np.sqrt(1.0 / self.rng.gamma(
            1.0, 1.0 / (1.0 / nu + lam2 / (2.0 * self.tau_global ** 2))))
        zeta = 1.0 / self.rng.gamma(1.0, 1.0 / (1.0 + 1.0 / self.tau_global ** 2))
        shape = 0.5 * (self.J * self.R + 1.0)
        rate = 1.0 / zeta + 0.5 * (lam2 / self.xi ** 2).sum()
        self.tau_global = float(np.sqrt(1.0 / self.rng.gamma(shape, 1.0 / rate)))
        self.xi = np.clip(self.xi, 1e-6, 1e6)
        self.tau_global = float(np.clip(self.tau_global, 1e-6, 1e3))

    # ----------------------------------------------------------------------- run
    def run(self, n_iter: int, n_burn: int, thin: int = 5,
            verbose_every: int = 50) -> Dict[str, np.ndarray]:
        keep_mu, keep_tau, keep_lam = [], [], []
        t0 = time.time()
        for it in range(n_iter):
            self.step()
            if it >= n_burn and (it - n_burn) % thin == 0:
                keep_mu.append((self.factors @ self.loadings.T).astype(np.float32))
                keep_tau.append(self.tau.astype(np.float32))
                keep_lam.append(self.loadings.astype(np.float32))
            if verbose_every and (it + 1) % verbose_every == 0:
                acc = sum(e.accept for e in self.ensembles)
                pro = max(sum(e.propose for e in self.ensembles), 1)
                print(f"[pg-bart] iter {it+1}/{n_iter} "
                      f"accept={acc/pro:.3f} tau_med={np.median(self.tau):.3f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        return {"residual_mean": np.stack(keep_mu) if keep_mu else np.zeros((0,)),
                "tau": np.stack(keep_tau) if keep_tau else np.zeros((0,)),
                "loadings": np.stack(keep_lam) if keep_lam else np.zeros((0,))}


def posterior_presence(draws: Dict[str, np.ndarray], offset: np.ndarray,
                       s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Posterior mean of sigmoid(theta) over retained draws, equation (31)."""
    res = draws["residual_mean"]
    taus = draws["tau"]
    if res.size == 0:
        raise ValueError("no retained posterior draws")
    acc = np.zeros(offset.shape, dtype=np.float64)
    for s in range(res.shape[0]):
        tau2_inv = 1.0 / np.maximum(taus[s].astype(np.float64) ** 2, 1e-8)
        mu = offset + res[s].astype(np.float64)
        var = 1.0 / (tau2_inv[None, :] + s2)
        mean = var * (mu * tau2_inv[None, :] + s1)
        acc += 1.0 / (1.0 + np.exp(-mean / np.sqrt(1.0 + math.pi * var / 8.0)))
    return acc / res.shape[0]
