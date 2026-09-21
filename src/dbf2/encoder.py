"""Module I: mass-defect-preserving peak-set encoder (specification section 3).

The spectrum is a set of peaks whose m/z is kept at full precision. The Fourier
phase is computed in float64 and reduced modulo 2*pi before the sine and cosine,
because at a wavelength of 2 mDa and m/z 1000 the raw phase exceeds 3e6 and
float32 would destroy exactly the mass-defect information the module exists to
preserve.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

ADDUCTS = ("[M+H]+", "[M-H]-", "[2M+Na]+", "[M+Na]+", "[2M+H]+", "[M+CH2O2-H]-",
           "[2M-H]-", "[M+NH4]+", "[M+Cl]-", "[M-H2O+H]+", "[M+K]+", "[M]+",
           "[M+C2H4O2-H]-", "[M-2H2O+H]+", "[M+2H]2+", "[2M+CH2O2-H]-")
POLARITIES = ("positive", "negative")
INSTRUMENTS = ("timsTOF", "Orbitrap", "QTOF", "other", "unknown")

ADDUCT_IDX = {a: i for i, a in enumerate(ADDUCTS)}
POLARITY_IDX = {p: i for i, p in enumerate(POLARITIES)}
INSTRUMENT_IDX = {p: i for i, p in enumerate(INSTRUMENTS)}
N_ADDUCT, N_POLARITY, N_INSTRUMENT = len(ADDUCTS) + 1, len(POLARITIES) + 1, len(INSTRUMENTS) + 1


class FourierMZ(nn.Module):
    """Multi-scale sinusoidal embedding, specification equation (9)."""

    def __init__(self, n_scales: int, lambda_min: float, lambda_max: float):
        super().__init__()
        lam = torch.logspace(math.log10(lambda_min), math.log10(lambda_max),
                             n_scales, dtype=torch.float64)
        self.register_buffer("inv_lambda", (2.0 * math.pi / lam), persistent=True)
        self.out_dim = 2 * n_scales

    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        # mz arrives as float64; reduce the phase before casting down
        phase = mz.to(torch.float64).unsqueeze(-1) * self.inv_lambda
        phase = torch.remainder(phase, 2.0 * math.pi).to(torch.float32)
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class EnergyBasis(nn.Module):
    """Smooth basis in harmonised collision energy, specification equation (12)."""

    def __init__(self, n_knots: int, scale: float = 100.0, e_max: float = 120.0):
        super().__init__()
        self.scale = scale
        centres = torch.linspace(0.0, e_max, n_knots)
        self.register_buffer("centres", centres, persistent=True)
        self.width = float(centres[1] - centres[0]) if n_knots > 1 else 1.0
        self.out_dim = n_knots + 3

    def forward(self, ce: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        e = torch.nan_to_num(ce, nan=0.0)
        z = (e / self.scale).unsqueeze(-1)
        rbf = torch.exp(-0.5 * ((e.unsqueeze(-1) - self.centres) / self.width) ** 2)
        rbf = rbf * (1.0 - missing.float()).unsqueeze(-1)
        return torch.cat([torch.ones_like(z), z, z ** 2, rbf], dim=-1)


class CovariateEncoder(nn.Module):
    """Acquisition covariates -> c_iv, specification equation (12)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_cov
        self.adduct = nn.Embedding(N_ADDUCT, d)
        self.polarity = nn.Embedding(N_POLARITY, d)
        self.instrument = nn.Embedding(N_INSTRUMENT, d)
        self.energy = EnergyBasis(cfg.energy_spline_knots, cfg.energy_scale)
        self.proj = nn.Sequential(
            nn.Linear(3 * d + self.energy.out_dim + 2, d), nn.GELU(), nn.Linear(d, d))
        self.out_dim = d

    def forward(self, batch: dict) -> torch.Tensor:
        e = self.energy(batch["ce_ev"], batch["ce_missing"])
        logmz = torch.log1p(batch["precursor_mz"].float()).unsqueeze(-1) / 10.0
        feats = torch.cat([
            self.adduct(batch["adduct_id"]),
            self.polarity(batch["polarity_id"]),
            self.instrument(batch["instrument_id"]),
            e, logmz, batch["ce_missing"].float().unsqueeze(-1)], dim=-1)
        return self.proj(feats)


class FiLMBlock(nn.Module):
    """Pre-norm Transformer block whose normalisation is modulated by the condition.

    Attention goes through ``scaled_dot_product_attention`` rather than
    ``nn.MultiheadAttention``. A spectrum can carry hundreds of peaks and a batch
    holds (molecules x views) of them, so materialising an explicit K x K matrix per
    head exhausts memory: 32 molecules, 4 views, 257 tokens and 8 heads is already
    270 MB per layer before autograd saves anything. The fused kernel keeps that
    off the heap.
    """

    def __init__(self, d: int, n_heads: int, d_ff: int, d_cov: int, dropout: float,
                 attention: bool = True):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"d_model {d} must be divisible by n_heads {n_heads}")
        self.n_heads, self.d_head = n_heads, d // n_heads
        self.attention = attention
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False) if attention else None
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False)
        if attention:
            self.qkv = nn.Linear(d, 3 * d)
            self.out = nn.Linear(d, d)
        self.p_attn = dropout
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.film = nn.Linear(d_cov, (4 if attention else 2) * d)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.drop = nn.Dropout(dropout)

    def _attend(self, h: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        n, k, d = h.shape
        q, kk, v = self.qkv(h).chunk(3, dim=-1)
        shape = (n, k, self.n_heads, self.d_head)
        q = q.view(shape).transpose(1, 2)
        kk = kk.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        a = F.scaled_dot_product_attention(
            q, kk, v, attn_mask=keep[:, None, None, :],
            dropout_p=self.p_attn if self.training else 0.0)
        return self.out(a.transpose(1, 2).reshape(n, k, d))

    def forward(self, x: torch.Tensor, cov: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        f = self.film(cov).unsqueeze(1)
        if self.attention:
            g1, b1, g2, b2 = f.chunk(4, dim=-1)
            h = self.norm1(x) * (1 + g1) + b1
            x = x + self.drop(self._attend(h, ~pad_mask))
        else:
            g2, b2 = f.chunk(2, dim=-1)
        h = self.norm2(x) * (1 + g2) + b2
        return x + self.drop(self.ff(h))


class PeakSetEncoder(nn.Module):
    """Peak set + acquisition condition -> H_iv, specification equation (13)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.fourier = FourierMZ(cfg.fourier_scales, cfg.lambda_min, cfg.lambda_max)
        d_tok = 2 * self.fourier.out_dim + 5
        self.token = nn.Sequential(nn.Linear(d_tok, cfg.d_model), nn.GELU(),
                                   nn.Linear(cfg.d_model, cfg.d_model))
        self.cls = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([
            FiLMBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.d_cov, cfg.dropout,
                      attention=cfg.attention)
            for _ in range(cfg.n_blocks)])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.out_dim = cfg.d_model

    def forward(self, mz: torch.Tensor, intensity: torch.Tensor,
                precursor: torch.Tensor, peak_mask: torch.Tensor,
                cov: torch.Tensor) -> torch.Tensor:
        """mz float64 (N, K); intensity (N, K); precursor float64 (N,);
        peak_mask bool (N, K) True where a real peak sits; cov (N, d_cov)."""
        prec = precursor.to(torch.float64)
        q = float(getattr(self.cfg, "quantise_mz", 0.0) or 0.0)
        if q > 0:
            # Under the rounding control the peak axis is already on the grid.
            # The precursor must be snapped to the same grid, or the neutral
            # loss would smuggle the precursor's full-precision mass defect
            # back into every token.
            prec = (torch.floor(prec / q) + 0.5) * q
        loss = prec.unsqueeze(-1) - mz.to(torch.float64)
        defect = (mz - torch.round(mz)).to(torch.float32)
        inten = intensity.float()
        rank = torch.cumsum(peak_mask.float(), dim=-1) / peak_mask.sum(-1, keepdim=True).clamp(min=1)
        feats = torch.cat([
            self.fourier(mz),
            self.fourier(loss.clamp(min=0.0)),
            defect.unsqueeze(-1),
            torch.sqrt(inten).unsqueeze(-1),
            torch.log1p(inten).unsqueeze(-1),
            rank.unsqueeze(-1),
            (loss <= 0).to(torch.float32).unsqueeze(-1),
        ], dim=-1)
        x = self.token(feats) * peak_mask.unsqueeze(-1).float()
        if self.cfg.attention:
            x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
            pad = torch.cat([torch.zeros_like(peak_mask[:, :1]), ~peak_mask], dim=1)
            # a view with no usable peaks would give an all-masked row; keep cls live
            pad[:, 0] = False
            for blk in self.blocks:
                x = blk(x, cov, pad)
            return self.norm(x[:, 0])
        # No attention: tokens never exchange information, so a CLS token would
        # only ever see itself. Pool the peak tokens by a masked mean instead.
        pad = ~peak_mask
        for blk in self.blocks:
            x = blk(x, cov, pad)
        m = peak_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return self.norm(pooled)


class BinnedEncoder(nn.Module):
    """Version-1 representation, retained only to reproduce ablation rung M0."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_bins = int(cfg.bin_max_mz / cfg.bin_width)
        self.net = nn.Sequential(
            nn.Linear(2 * self.n_bins, cfg.d_model * 2), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model * 2, cfg.d_model))
        self.out_dim = cfg.d_model

    def forward(self, mz, intensity, precursor, peak_mask, cov):
        n, k = mz.shape
        w, nb = self.cfg.bin_width, self.n_bins
        frag = torch.zeros(n, nb, device=mz.device, dtype=torch.float32)
        loss = torch.zeros(n, nb, device=mz.device, dtype=torch.float32)
        val = torch.sqrt(intensity.float()) * peak_mask.float()
        bi = torch.clamp((mz.float() / w).long(), 0, nb - 1)
        bl = torch.clamp(((precursor.float().unsqueeze(-1) - mz.float()) / w).long(), 0, nb - 1)
        frag.scatter_add_(1, bi, val)
        loss.scatter_add_(1, bl, val)
        x = torch.cat([frag, loss], dim=-1)
        return self.net(F.normalize(x, dim=-1))


def build_encoder(cfg: ModelConfig) -> nn.Module:
    if cfg.encoder == "peakset":
        return PeakSetEncoder(cfg)
    if cfg.encoder == "binned":
        return BinnedEncoder(cfg)
    raise ValueError(f"unknown encoder {cfg.encoder!r}")
