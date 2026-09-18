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
