"""The rounding control (rung M1R).

These tests exist to defend one claim: that M1R differs from M1 in the
precision of the mass axis and in nothing else. Each test pins down one way
that claim could quietly be false.
"""
import numpy as np
import pytest
import torch

from dbf2.config import Config
from dbf2.dataset import quantise_merge


W = 0.5


def test_disabled_is_identity():
    mz = np.array([67.0178, 67.0542, 121.0648])
    it = np.array([0.4, 0.9, 1.0])
    for width in (0.0, None):
        q, qi = quantise_merge(mz, it, width)
        assert np.array_equal(q, mz) and np.array_equal(qi, it)


def test_peaks_land_on_bin_centres():
    mz = np.array([67.0178, 121.0648, 300.9])
    q, _ = quantise_merge(mz, np.ones(3), W)
    assert np.allclose(q, (np.floor(mz / W) + 0.5) * W)


def test_bin_index_matches_binned_encoder():
    """BinnedEncoder indexes with long(mz / w), i.e. floor for positive mz.
    The control must use the same cells or it is not the same discretisation."""
    mz = np.array([0.1, 0.49, 0.5, 67.0178, 999.75])
    enc_idx = (torch.tensor(mz) / W).long().numpy()
    ctl_idx = np.floor(mz / W).astype(np.int64)
    assert np.array_equal(enc_idx, ctl_idx)


def test_isobars_in_one_bin_become_indistinguishable():
    """C4H3O+ and C5H7+ differ by 36 mDa and share a 0.5 Da bin. After the
    control they must be a single peak: this is the information the paper
    claims binning destroys."""
    mz = np.array([67.0178, 67.0542])
    q, qi = quantise_merge(mz, np.array([1.0, 1.0]), W)
    assert q.size == 1 and qi.size == 1
    assert q[0] == pytest.approx(67.25)


def test_merge_sqrt_preserves_binned_accumulator():
    """BinnedEncoder accumulates sqrt intensity. The merged peak must carry the
    same total, so the control differs from M0 in encoder only, not in mass."""
    it = np.array([0.25, 0.81])
    _, qi = quantise_merge(np.array([67.01, 67.05]), it, W, mode="sqrt")
    assert np.sqrt(qi[0]) == pytest.approx(np.sqrt(it).sum())


@pytest.mark.parametrize("mode,expected", [("sum", 1.06), ("max", 0.81)])
def test_other_merge_modes(mode, expected):
    _, qi = quantise_merge(np.array([67.01, 67.05]), np.array([0.25, 0.81]), W, mode)
    assert qi[0] == pytest.approx(expected)


def test_unknown_merge_mode_rejected():
    with pytest.raises(ValueError):
        quantise_merge(np.array([1.0, 1.1]), np.array([1.0, 1.0]), W, "mean")


def test_output_stays_sorted_and_deduplicated():
    rng = np.random.RandomState(0)
    mz = np.sort(rng.uniform(50, 400, 500))
    q, qi = quantise_merge(mz, rng.uniform(0, 1, 500), W)
    assert q.size == qi.size
    assert np.all(np.diff(q) > 0)
    assert q.size == np.unique(np.floor(mz / W)).size


def test_mass_defect_is_destroyed():
    """After the control the defect channel can only report which half of the
    nominal mass a peak fell in -- exactly what the bin index already says."""
    rng = np.random.RandomState(1)
    mz = rng.uniform(50, 400, 2000)
    q, _ = quantise_merge(mz, np.ones(2000), W)
    defect = q - np.round(q)
    assert set(np.unique(np.round(defect, 6))) <= {0.25, -0.25}


def test_absorption_is_measured_not_assumed():
    """The merge must actually remove peaks when they collide."""
    rng = np.random.RandomState(2)
    mz = np.sort(rng.uniform(50, 400, 300))
    q, _ = quantise_merge(mz, np.ones(300), W)
    absorbed = 1.0 - q.size / mz.size
    assert absorbed > 0.05


def test_config_M1R_matches_M1_except_quantisation():
    a = Config.ablation("M1", ".").model
    b = Config.ablation("M1R", ".").model
    assert b.quantise_mz == b.bin_width > 0
    assert a.quantise_mz == 0.0
    differing = {k for k in vars(a) if getattr(a, k) != getattr(b, k)}
    assert differing == {"quantise_mz"}, differing


def test_M1R_has_identical_parameter_count_to_M1():
    """If the control changed capacity it would reintroduce the confound it
    exists to remove."""
    from dbf2.model import DeepBayesFrag
    n = []
    for rung in ("M1", "M1R"):
        cfg = Config.ablation(rung, ".")
        torch.manual_seed(0)
        n.append(sum(p.numel() for p in DeepBayesFrag(cfg.model).parameters()))
    assert n[0] == n[1]


def test_encoder_rounds_the_precursor():
    """Neutral loss = precursor - mz. With mz on the grid but the precursor at
    full precision, the precursor's defect would leak into every token."""
    from dbf2.encoder import PeakSetEncoder
    cfg = Config.ablation("M1R", ".").model
    enc = PeakSetEncoder(cfg).eval()      # dropout off: compare the mapping, not a sample
    mz = torch.tensor([[67.25, 121.25]], dtype=torch.float64)
    mask = torch.ones(1, 2, dtype=torch.bool)
    cov = torch.zeros(1, cfg.d_cov)
    inten = torch.ones(1, 2)
    a = enc(mz, inten, torch.tensor([342.1617], dtype=torch.float64), mask, cov)
    b = enc(mz, inten, torch.tensor([342.4988], dtype=torch.float64), mask, cov)
    # both precursors fall in the same 0.5 Da cell, so the control must make
    # them produce the same embedding
    assert torch.allclose(a, b, atol=1e-6)

    # and that without the control the same two precursors would differ,
    # which is the leak this rounding exists to close
    cfg_free = Config.ablation("M1", ".").model
    torch.manual_seed(0)
    free = PeakSetEncoder(cfg_free).eval()
    c = free(mz, inten, torch.tensor([342.1617], dtype=torch.float64), mask, cov)
    d = free(mz, inten, torch.tensor([342.4988], dtype=torch.float64), mask, cov)
    assert not torch.allclose(c, d, atol=1e-6)


# --------------------------------------------------------------- rung M1D
# M1D removes self-attention while holding the mass axis and the token interface
# fixed against M1R, so that M1D - M0 prices the dense-to-sparse switch and
# M1R - M1D prices attention. These tests pin down that it changes only that.

def test_M1D_differs_from_M1R_only_in_attention_and_width():
    a = Config.ablation("M1R", ".").model
    b = Config.ablation("M1D", ".").model
    differing = {k for k in vars(a) if getattr(a, k) != getattr(b, k)}
    assert differing == {"attention", "d_ff"}, differing
    assert a.quantise_mz == b.quantise_mz > 0      # same binned mass axis
    assert b.attention is False


def test_M1D_is_not_the_smaller_arm():
    """A control that also had fewer parameters would reintroduce the confound
    it exists to remove."""
    from dbf2.model import DeepBayesFrag
    n = {}
    for rung in ("M1D", "M1R"):
        cfg = Config.ablation(rung, ".")
        cfg.model.n_presence, cfg.model.n_count = 512, 88
        torch.manual_seed(0)
        n[rung] = sum(p.numel() for p in DeepBayesFrag(cfg.model).parameters())
    assert n["M1D"] >= n["M1R"], n


def test_M1D_blocks_carry_no_attention_weights():
    from dbf2.encoder import PeakSetEncoder
    enc = PeakSetEncoder(Config.ablation("M1D", ".").model)
    for blk in enc.blocks:
        assert not hasattr(blk, "qkv") and not hasattr(blk, "out")
        assert blk.attention is False


def test_M1D_tokens_do_not_exchange_information():
    """The point of the arm. Run the block stack directly on token embeddings:
    with attention removed, a token's output must not depend on what other
    tokens are present. With attention on, it must."""
    from dbf2.encoder import FiLMBlock
    cfg = Config.ablation("M1D", ".").model
    torch.manual_seed(0)
    blk = FiLMBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.d_cov, 0.0,
                    attention=cfg.attention).eval()
    cov = torch.zeros(1, cfg.d_cov)
    torch.manual_seed(1)
    x = torch.randn(1, 3, cfg.d_model)
    pad = torch.zeros(1, 3, dtype=torch.bool)
    both = blk(x, cov, pad)
    alone = blk(x[:, :1], cov, pad[:, :1])
    assert torch.allclose(both[:, 0], alone[:, 0], atol=1e-6)

    acfg = Config.ablation("M1R", ".").model
    torch.manual_seed(0)
    ablk = FiLMBlock(acfg.d_model, acfg.n_heads, acfg.d_ff, acfg.d_cov, 0.0,
                     attention=True).eval()
    ax = torch.randn(1, 3, acfg.d_model)
    a_both = ablk(ax, cov, pad)
    a_alone = ablk(ax[:, :1], cov, pad[:, :1])
    assert not torch.allclose(a_both[:, 0], a_alone[:, 0], atol=1e-6)


def test_M1D_pools_peaks_rather_than_a_cls_token():
    """A CLS token that attends to nothing would learn nothing, so the encoder
    must fall back to a masked mean over the peak tokens."""
    from dbf2.encoder import PeakSetEncoder
    cfg = Config.ablation("M1D", ".").model
    enc = PeakSetEncoder(cfg).eval()
    mz = torch.tensor([[67.25, 121.25]], dtype=torch.float64)
    h = enc(mz, torch.ones(1, 2), torch.tensor([342.25], dtype=torch.float64),
            torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, cfg.d_cov))
    assert h.shape == (1, cfg.d_model) and torch.isfinite(h).all()


# ------------------------------------------- acquisition covariates reach every arm
# BinnedEncoder once accepted `cov` and ignored it, so M0 alone could not see the
# collision energy, adduct, polarity or instrument while every peak-set arm was
# conditioned on them. That silently folded "the baseline is blind to the
# acquisition" into every contrast measured against M0. These tests keep the
# pathway open in all arms.

def _batch(ce=10.0, adduct=0, instrument=0, seed=0):
    torch.manual_seed(seed)
    B, V, K = 2, 2, 8
    return dict(
        mz=torch.rand(B, V, K, dtype=torch.float64) * 300 + 50,
        intensity=torch.rand(B, V, K),
        precursor_mz=torch.full((B, V), 400.0, dtype=torch.float64),
        peak_mask=torch.ones(B, V, K, dtype=torch.bool),
        view_mask=torch.ones(B, V, dtype=torch.bool),
        adduct_id=torch.full((B, V), adduct, dtype=torch.long),
        polarity_id=torch.zeros(B, V, dtype=torch.long),
        instrument_id=torch.full((B, V), instrument, dtype=torch.long),
        ce_ev=torch.full((B, V), ce), ce_missing=torch.zeros(B, V))


@pytest.mark.parametrize("rung", ["M0", "M1D", "M1R", "M1", "M2"])
def test_every_arm_responds_to_the_acquisition(rung):
    """Same spectrum, different collision energy, adduct and instrument: the
    prediction must change. FiLM is zero-initialised, so the conditioning path is
    inactive at step 0 by design; perturb it first or the test proves nothing."""
    import torch.nn as nn
    from dbf2.model import DeepBayesFrag
    cfg = Config.ablation(rung, ".")
    cfg.model.n_presence, cfg.model.n_count = 512, 88
    torch.manual_seed(0)
    m = DeepBayesFrag(cfg.model)
    for n, p in m.named_parameters():
        if ".film." in n:
            nn.init.normal_(p, std=0.05)
    m.eval()
    a = m(_batch(ce=10.0, adduct=0, instrument=0))["logit"]
    b = m(_batch(ce=90.0, adduct=1, instrument=2))["logit"]
    assert not torch.allclose(a, b, atol=1e-6), f"{rung} ignores the acquisition"


@pytest.mark.parametrize("rung", ["M0", "M1D", "M1R", "M1", "M2"])
def test_covariate_encoder_is_reachable(rung):
    """A gradient must reach cov_encoder, or its weights are dead."""
    import torch.nn as nn
    from dbf2.model import DeepBayesFrag
    cfg = Config.ablation(rung, ".")
    cfg.model.n_presence, cfg.model.n_count = 512, 88
    torch.manual_seed(0)
    m = DeepBayesFrag(cfg.model)
    for n, p in m.named_parameters():
        if ".film." in n:
            nn.init.normal_(p, std=0.05)
    m(_batch())["logit"].sum().backward()
    g = sum(p.grad.abs().sum().item() for n, p in m.named_parameters()
            if n.startswith("cov_encoder") and p.grad is not None)
    assert g > 0.0, f"{rung} leaves cov_encoder without gradient"


def test_binned_encoder_sees_covariates():
    """The specific regression: BinnedEncoder must consume cov, not just accept it."""
    from dbf2.encoder import BinnedEncoder
    cfg = Config.ablation("M0", ".").model
    enc = BinnedEncoder(cfg).eval()
    mz = torch.rand(2, 8, dtype=torch.float64) * 300 + 50
    inten = torch.rand(2, 8)
    prec = torch.full((2,), 400.0, dtype=torch.float64)
    mask = torch.ones(2, 8, dtype=torch.bool)
    torch.manual_seed(1)
    c1 = torch.randn(2, cfg.d_cov)
    c2 = torch.randn(2, cfg.d_cov)
    assert not torch.allclose(enc(mz, inten, prec, mask, c1),
                              enc(mz, inten, prec, mask, c2), atol=1e-6)


def test_M0_remains_the_largest_arm():
    """M0 gains a covariate input; it must still not be handicapped on capacity."""
    from dbf2.model import DeepBayesFrag
    n = {}
    for rung in ("M0", "M1D", "M1R", "M1", "M2"):
        cfg = Config.ablation(rung, ".")
        cfg.model.n_presence, cfg.model.n_count = 512, 88
        torch.manual_seed(0)
        n[rung] = sum(p.numel() for p in DeepBayesFrag(cfg.model).parameters())
    assert n["M0"] == max(n.values()), n
