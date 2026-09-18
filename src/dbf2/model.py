"""Assembly of Modules I, II and the count head (specification sections 3-5)."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .encoder import build_encoder, CovariateEncoder
from .measurement import MeasurementModel, MeanPooling, pool_representation


class NegBinHead(nn.Module):
    """Count targets, specification equation (29)."""

    def __init__(self, d_h: int, n_count: int):
        super().__init__()
        self.mean = nn.Linear(d_h, n_count)
        self.log_phi = nn.Parameter(torch.zeros(n_count))

    def forward(self, h_pool: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"log_mu": self.mean(h_pool).clamp(-8.0, 8.0),
                "log_phi": self.log_phi.clamp(-4.0, 6.0)}

    @staticmethod
    def nll(log_mu: torch.Tensor, log_phi: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu, phi = torch.exp(log_mu), torch.exp(log_phi)
        y = y.float()
        return -(torch.lgamma(y + phi) - torch.lgamma(phi) - torch.lgamma(y + 1.0)
                 + phi * (log_phi - torch.log(phi + mu))
                 + y * (log_mu - torch.log(phi + mu)))


class DeepBayesFrag(nn.Module):
    """X -> H_iv -> pooled structural posterior."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.cov_encoder = CovariateEncoder(cfg)
        self.encoder = build_encoder(cfg)
        d_h = self.encoder.out_dim
        self.d_h = d_h
        if cfg.pooling == "hierarchical":
            self.aggregator = MeasurementModel(cfg, d_h, cfg.n_presence)
        elif cfg.pooling == "mean":
            self.aggregator = MeanPooling(cfg, d_h, cfg.n_presence)
        else:
            raise ValueError(f"unknown pooling {cfg.pooling!r}")
        self.count_head = NegBinHead(d_h, cfg.n_count) if cfg.count_head else None
        self.proj_spectrum = (nn.Linear(d_h, cfg.d_proj) if cfg.contrastive else None)

    # ------------------------------------------------------------------- encoding
    def encode_views(self, batch: Dict[str, torch.Tensor]):
        """Flatten (B, V) to a single batch of views, encode, and restore the shape."""
        b, v = batch["view_mask"].shape
        flat = {k: batch[k].reshape(b * v, *batch[k].shape[2:])
                for k in ("adduct_id", "polarity_id", "instrument_id",
                          "ce_ev", "ce_missing", "precursor_mz")}
        cov_flat = self.cov_encoder(flat)
        h_flat = self.encoder(
            batch["mz"].reshape(b * v, -1), batch["intensity"].reshape(b * v, -1),
            batch["precursor_mz"].reshape(b * v), batch["peak_mask"].reshape(b * v, -1),
            cov_flat)
        return h_flat.view(b, v, -1), cov_flat.view(b, v, -1)

    def forward(self, batch: Dict[str, torch.Tensor],
                prior_offset: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h_view, cov = self.encode_views(batch)
        mask = batch["view_mask"]

        if isinstance(self.aggregator, MeasurementModel):
            with torch.no_grad() if not self.training else torch.enable_grad():
                w = self.aggregator.view_precision(cov)        # (B, V, J)
            h_pool = pool_representation(h_view, mask, w.mean(dim=-1))
        else:
            h_pool = pool_representation(h_view, mask)

        out = self.aggregator(h_view, cov, mask, h_pool, prior_offset)
        out["h_pool"] = h_pool
        out["h_view"] = h_view
        out["cov"] = cov
        if self.count_head is not None:
            out.update({f"count_{k}": v for k, v in self.count_head(h_pool).items()})
        if self.proj_spectrum is not None:
            out["z_spectrum"] = F.normalize(self.proj_spectrum(h_pool), dim=-1)
        return out


class FingerprintEncoder(nn.Module):
    """Structure side of the contrastive objective, specification equation (36).

    A Morgan-fingerprint multilayer perceptron stands in for a graph network; it is
    cheap, and the contrastive term is an auxiliary objective rather than the claim.
    """

    def __init__(self, n_bits: int, d_hidden: int, d_proj: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_bits, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_proj))

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(fp.float()), dim=-1)


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z_a @ z_b.t() / temperature
    target = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
