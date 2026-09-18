"""Module II: hierarchical multi-view measurement model (specification section 4).

Each view is a noisy, condition-dependent measurement of a shared latent structural
state,

    l_ivj | theta_ij, C_iv  ~  N( alpha_j(c) + beta_j(c) theta_ij , sigma_j(c)^2 ),

and the views combine conjugately into

    v_ij^-1 = tau_ij^-2 + sum_v w_ivj,     w_ivj = beta_j(c)^2 / sigma_j(c)^2,
    m_ij    = v_ij ( mu_ij tau_ij^-2 + sum_v w_ivj ( l_ivj - alpha_j(c) ) / beta_j(c) ).

The sufficient statistics are accumulated as

    S1 = sum_v beta (l - alpha) / sigma^2       (equals sum_v w * l~)
    S2 = sum_v beta^2 / sigma^2                 (equals sum_v w)

so that no division by a possibly small beta ever appears, which is the only
numerically delicate step in the update.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

_PI_OVER_8 = math.pi / 8.0


class MeasurementModel(nn.Module):
    """Per-view evidence, condition-dependent precision, conjugate pooling."""

    def __init__(self, cfg: ModelConfig, d_h: int, n_targets: int):
        super().__init__()
        self.cfg = cfg
        self.n_targets = n_targets

        # per-view evidence logits, equation (14)
        self.evidence = nn.Linear(d_h, n_targets)

        # condition-dependent measurement parameters, equations (16)-(18)
        d_c = cfg.d_cov
        self.alpha0 = nn.Parameter(torch.zeros(n_targets))
        self.beta0 = nn.Parameter(torch.full((n_targets,), 0.5413))   # softplus -> ~1
        self.logsig0 = nn.Parameter(torch.zeros(n_targets))
        self.alpha_c = nn.Parameter(torch.zeros(n_targets, d_c))
        self.beta_c = nn.Parameter(torch.zeros(n_targets, d_c))
        self.logsig_c = nn.Parameter(torch.zeros(n_targets, d_c))

        # prior on theta, equation (25); the BART residual is added in stage 2
        self.prior_mean = nn.Linear(d_h, n_targets)
        self.log_tau = nn.Parameter(torch.zeros(n_targets),
                                    requires_grad=cfg.learn_prior_scale)

    # ------------------------------------------------------------------ parameters
    def condition_params(self, cov: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """cov (B, V, d_cov) -> alpha, beta, sigma each (B, V, J)."""
        alpha = self.alpha0 + cov @ self.alpha_c.t()
        beta = F.softplus(self.beta0 + cov @ self.beta_c.t()) + self.cfg.beta_floor
        logsig = (self.logsig0 + cov @ self.logsig_c.t()).clamp(
            -self.cfg.log_sigma_clamp, self.cfg.log_sigma_clamp)
        return alpha, beta, torch.exp(logsig)

    def view_precision(self, cov: torch.Tensor) -> torch.Tensor:
        """w_ivj = beta^2 / sigma^2, clamped. Depends on the condition only, so it can
        be computed before the pooled representation exists."""
        _, beta, sigma = self.condition_params(cov)
        return ((beta / sigma) ** 2).clamp(max=self.cfg.max_precision)

    # ---------------------------------------------------------------------- pooling
    def forward(self, h_view: torch.Tensor, cov: torch.Tensor, view_mask: torch.Tensor,
                h_pool: torch.Tensor, prior_offset: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        h_view      (B, V, d_h)   per-view representations
        cov         (B, V, d_cov) acquisition covariates
        view_mask   (B, V) bool   True where a real view sits
        h_pool      (B, d_h)      precision-pooled representation, equation (23)
        prior_offset(B, J)        optional BART residual term from stage 2
        """
        ell = self.evidence(h_view)                            # (B, V, J)
        alpha, beta, sigma = self.condition_params(cov)
        inv_var = 1.0 / (sigma ** 2)
        m = view_mask.unsqueeze(-1).to(ell.dtype)

        s1 = (beta * (ell - alpha) * inv_var * m).sum(dim=1)    # (B, J)
        s2 = ((beta ** 2) * inv_var * m).sum(dim=1).clamp(max=self.cfg.max_precision)

        mu = self.prior_mean(h_pool)
        if prior_offset is not None:
            mu = mu + prior_offset
        tau2_inv = torch.exp(-2.0 * self.log_tau)

        var = 1.0 / (tau2_inv + s2)
        mean = var * (mu * tau2_inv + s1)
        logit = mean / torch.sqrt(1.0 + _PI_OVER_8 * var)        # equation (22)
        return {"logit": logit, "theta_mean": mean, "theta_var": var,
                "prior_mean": mu, "s1": s1, "s2": s2, "ell": ell,
                "beta": beta, "sigma": sigma}


class MeanPooling(nn.Module):
    """Ablation rung M1: mean over views, no measurement model."""

    def __init__(self, cfg: ModelConfig, d_h: int, n_targets: int):
        super().__init__()
        self.head = nn.Linear(d_h, n_targets)

    def forward(self, h_view, cov, view_mask, h_pool, prior_offset=None):
        logit = self.head(h_pool)
        if prior_offset is not None:
            logit = logit + prior_offset
        zero = torch.zeros_like(logit)
        return {"logit": logit, "theta_mean": logit, "theta_var": zero,
                "prior_mean": logit, "s1": zero, "s2": zero,
                "ell": None, "beta": None, "sigma": None}


def pool_representation(h_view: torch.Tensor, view_mask: torch.Tensor,
                        weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Equation (23). ``weight`` is (B, V); when absent this is a masked mean."""
    m = view_mask.to(h_view.dtype)
    w = m if weight is None else weight * m
    denom = w.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (h_view * w.unsqueeze(-1)).sum(dim=1) / denom


@torch.no_grad()
def accumulate_statistics(model: "MeasurementModel", h_view: torch.Tensor,
                          cov: torch.Tensor, view_mask: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """S1 and S2 for a chunk of views.

    Because the pooling in equation (20) is a sum over views, a molecule with
    hundreds or thousands of spectra can be processed in chunks and the statistics
    added, giving exactly the same posterior as a single pass. This is what allows
    the specification's claim that no cap on the number of views is required.
    """
    ell = model.evidence(h_view)
    alpha, beta, sigma = model.condition_params(cov)
    inv_var = 1.0 / (sigma ** 2)
    m = view_mask.unsqueeze(-1).to(ell.dtype)
    s1 = (beta * (ell - alpha) * inv_var * m).sum(dim=1)
    s2 = ((beta ** 2) * inv_var * m).sum(dim=1)
    return s1, s2


def posterior_from_statistics(model: "MeasurementModel", s1: torch.Tensor,
                              s2: torch.Tensor, mu: torch.Tensor
                              ) -> Dict[str, torch.Tensor]:
    """Close the conjugate update from accumulated sufficient statistics."""
    tau2_inv = torch.exp(-2.0 * model.log_tau)
    s2 = s2.clamp(max=model.cfg.max_precision)
    var = 1.0 / (tau2_inv + s2)
    mean = var * (mu * tau2_inv + s1)
    return {"logit": mean / torch.sqrt(1.0 + _PI_OVER_8 * var),
            "theta_mean": mean, "theta_var": var}
