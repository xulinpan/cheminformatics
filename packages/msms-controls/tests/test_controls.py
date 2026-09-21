"""Tests for the control API.

These cover the two mistakes the controls exist to prevent: a rounding control
that leaks precision through the precursor, and an arm that silently ignores the
side information every other arm receives.
"""
import numpy as np
import pytest

from msms_controls import (ConditioningParityError, absorbed_fraction,
                          check_conditioning_parity, conditioning_response,
                          quantise_peaks, quantise_spectrum)

W = 0.5
# the isobaric pair the mass-defect argument turns on: same 0.5 Da bin
C4H3O, C5H7 = 67.0178, 67.0542


# ------------------------------------------------------------------ quantisation
def test_isobars_in_one_cell_become_one_peak():
    mz, inten = quantise_peaks([C4H3O, C5H7], [100.0, 40.0], W)
    assert mz.size == 1
    assert mz[0] == pytest.approx(67.25)          # the cell centre, not either peak


def test_peaks_land_on_cell_centres():
    mz, _ = quantise_peaks([10.01, 200.49, 200.51], [1.0, 1.0, 1.0], W)
    assert np.allclose(mz, [10.25, 200.25, 200.75])


def test_nothing_is_added_or_dropped():
    rng = np.random.default_rng(0)
    raw = np.sort(rng.uniform(50, 800, 200))
    mz, inten = quantise_peaks(raw, np.ones_like(raw), W)
    assert mz.size == np.unique(np.floor(raw / W)).size
    assert mz.size == inten.size


def test_sqrt_merge_matches_a_binned_accumulator():
    """The default must reproduce what a sqrt-intensity binned vector accumulates."""
    _, inten = quantise_peaks([C4H3O, C5H7], [100.0, 36.0], W, merge="sqrt")
    assert inten[0] == pytest.approx((10.0 + 6.0) ** 2)


def test_sum_and_max_merges():
    _, s = quantise_peaks([C4H3O, C5H7], [100.0, 36.0], W, merge="sum")
    _, m = quantise_peaks([C4H3O, C5H7], [100.0, 36.0], W, merge="max")
    assert s[0] == pytest.approx(136.0)
    assert m[0] == pytest.approx(100.0)


def test_unknown_merge_mode_is_refused():
    with pytest.raises(ValueError, match="unknown merge mode"):
        quantise_peaks([1.0], [1.0], W, merge="mean")


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="differ in length"):
        quantise_peaks([1.0, 2.0], [1.0], W)


@pytest.mark.parametrize("width", [0.0, -1.0])
def test_non_positive_width_is_a_no_op(width):
    mz, inten = quantise_peaks([C4H3O, C5H7], [1.0, 2.0], width)
    assert np.allclose(mz, [C4H3O, C5H7])
    assert np.allclose(inten, [1.0, 2.0])


def test_empty_spectrum_survives():
    mz, inten = quantise_peaks([], [], W)
    assert mz.size == 0 and inten.size == 0


def test_full_precision_is_actually_discarded():
    """The whole point: after the control, the mass defect is gone."""
    mz, _ = quantise_peaks([C4H3O], [1.0], W)
    assert mz[0] != pytest.approx(C4H3O)
    assert (mz[0] / W) % 1 == pytest.approx(0.5)


# ------------------------------------------------------------- the precursor leak
def test_precursor_is_snapped_to_the_same_grid():
    _, _, prec = quantise_spectrum([100.0], [1.0], 300.1234, W)
    assert prec == pytest.approx(300.25)


def test_neutral_loss_carries_no_precursor_defect():
    """Two precursors in one cell must give identical neutral losses.

    This is the leak quantise_peaks alone would leave: an unsnapped precursor
    returns its own mass defect to every token through precursor - mz.
    """
    peaks, inten = [120.0], [1.0]
    _, _, pa = quantise_spectrum(peaks, inten, 300.1001, W)
    _, _, pb = quantise_spectrum(peaks, inten, 300.4999, W)
    qa, _, _ = quantise_spectrum(peaks, inten, 300.1001, W)
    qb, _, _ = quantise_spectrum(peaks, inten, 300.4999, W)
    assert (pa - qa[0]) == pytest.approx(pb - qb[0])
    # and without the control the same two precursors differ
    assert (300.1001 - 120.0) != pytest.approx(300.4999 - 120.0)


def test_quantise_spectrum_leaves_the_precursor_alone_at_zero_width():
    _, _, prec = quantise_spectrum([1.0], [1.0], 300.1234, 0.0)
    assert prec == pytest.approx(300.1234)


# ----------------------------------------------------------------- absorption
def test_absorbed_fraction_counts_merged_peaks():
    assert absorbed_fraction([C4H3O, C5H7], W) == pytest.approx(0.5)
    assert absorbed_fraction([10.1, 200.1, 400.1], W) == pytest.approx(0.0)


def test_absorption_falls_as_the_grid_gets_finer():
    rng = np.random.default_rng(1)
    mz = rng.uniform(50, 500, 500)
    coarse = absorbed_fraction(mz, 0.5)
    fine = absorbed_fraction(mz, 0.01)
    assert coarse > fine


def test_absorbed_fraction_degenerate_inputs():
    assert absorbed_fraction([], W) == 0.0
    assert absorbed_fraction([1.0, 2.0], 0.0) == 0.0


# ------------------------------------------------------- conditioning parity
LOW = {"collision_energy": 10.0, "adduct": 0}
HIGH = {"collision_energy": 90.0, "adduct": 1}


def _listening(cov):
    return np.array([cov["collision_energy"] * 0.01 + cov["adduct"]])


def _deaf(cov):
    return np.array([0.7])


def test_response_is_positive_for_a_conditioned_arm():
    assert conditioning_response(_listening, LOW, HIGH) > 0


def test_response_is_zero_for_an_unconditioned_arm():
    assert conditioning_response(_deaf, LOW, HIGH) == 0.0


def test_parity_passes_when_every_arm_listens():
    out = check_conditioning_parity({"M1": _listening, "M2": _listening}, LOW, HIGH)
    assert set(out) == {"M1", "M2"} and all(v > 0 for v in out.values())


def test_parity_names_the_arm_that_does_not():
    """The regression this exists for: one arm accepts the covariates and drops them."""
    with pytest.raises(ConditioningParityError) as e:
        check_conditioning_parity({"M0": _deaf, "M1": _listening}, LOW, HIGH)
    assert "M0" in str(e.value)
    assert "M1" not in str(e.value).split(":")[1].split(".")[0]


def test_parity_reports_every_failing_arm():
    with pytest.raises(ConditioningParityError) as e:
        check_conditioning_parity({"M0": _deaf, "M0b": _deaf, "M1": _listening},
                                  LOW, HIGH)
    assert "M0" in str(e.value) and "M0b" in str(e.value)
    assert "2 of 3" in str(e.value)


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="different shapes"):
        conditioning_response(lambda c: np.zeros(1 + c["adduct"]), LOW, HIGH)
