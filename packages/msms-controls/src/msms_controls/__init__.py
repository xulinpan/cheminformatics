"""Two controls for attributing gains in MS/MS representation studies.

A comparison between a binned baseline and a peak-level model changes several
things at once, and the increment belongs to the whole package unless the design
contains an intervention that moves one component. This package provides the two
interventions a controlled decomposition of one such comparison found necessary.
It depends on nothing but numpy: no models, no dataset layout, no trained
weights, no deep-learning framework. Apply it to whatever model you already
have.

**The rounding control.** A binned intensity vector and a peak set are different
kinds of object and no architecture reads both, so abandoning bins forces a change
of encoder and the two effects cannot be separated. They can be separated from the
other side: leave the peak-set interface alone and degrade the *mass axis* in
place, by snapping peaks to the bin grid the baseline would have used and merging
the ones that then collide. The resulting arm carries the baseline's mass
information through the treatment's architecture. Use :func:`quantise_spectrum`.

**The conditioning-parity check.** The same study found a confound larger than
anything it set out to measure: the binned baseline's encoder accepted the
acquisition covariates and discarded them, so it alone predicted without knowing
the collision energy, adduct, polarity or instrument. Supplying them closed half
the gap the comparison had been attributing to the spectral representation. The
failure is silent - the covariates are passed, the code runs, the numbers look
plausible - so it needs an explicit test. Use :func:`check_conditioning_parity`.

Both are cheap, and neither is specific to mass spectrometry. Wherever a change of
representation forces a change of architecture, ask whether the richer
representation can be impoverished in place; and wherever arms of an ablation are
given side information, check that each of them can actually see it.

Reference: Pan X, Wang C. A metadata-matched binned baseline halves the apparent
peak-level advantage in MS/MS substructure prediction.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence, Tuple

import numpy as np

__all__ = ["quantise_peaks", "quantise_spectrum", "absorbed_fraction",
           "conditioning_response", "check_conditioning_parity",
           "ConditioningParityError"]

_MERGE_MODES = ("sqrt", "sum", "max")


def quantise_peaks(mz, intensity, width: float, *, merge: str = "sqrt"
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Snap peaks to a bin grid and merge the ones that collide.

    Peaks are assigned to cells by ``floor(mz / width)`` - the indexing a binned
    encoder uses - and each surviving peak is placed at its cell centre. Peaks
    sharing a cell are merged into one.

    Parameters
    ----------
    mz, intensity : array-like, same length
        One spectrum's peaks. ``mz`` is read in float64; pass it at full
        precision, since the point of the control is to discard that precision
        deliberately rather than to have lost it earlier.
    width : float
        Grid width in the same unit as ``mz`` (Da for a 0.5 Da baseline).
        Non-positive width returns the input unchanged.
    merge : {"sqrt", "sum", "max"}
        How colliding peaks combine. ``"sqrt"`` accumulates square-root intensity
        and squares the total, which is what a binned vector built on square-root
        intensity accumulates; match this to whatever your baseline does, or the
        control silently changes the intensity scale as well as the mass axis.

    Returns
    -------
    (mz, intensity) : tuple of np.ndarray
        The peaks a full-precision encoder would receive if the mass axis had been
        discretised first. Sorted by ``mz``. Nothing else about the spectrum
        changes: no peaks are added or dropped, only merged.

    Notes
    -----
    For a whole spectrum prefer :func:`quantise_spectrum`, which also snaps the
    precursor. Quantising the peaks alone leaves a leak - see that function.
    """
    if merge not in _MERGE_MODES:
        raise ValueError(f"unknown merge mode {merge!r}; expected one of {_MERGE_MODES}")
    mz64 = np.asarray(mz, dtype=np.float64).ravel()
    it64 = np.asarray(intensity, dtype=np.float64).ravel()
    if mz64.shape != it64.shape:
        raise ValueError(f"mz and intensity differ in length: {mz64.size} vs {it64.size}")
    if not width or width <= 0 or mz64.size == 0:
        return mz64, it64

    idx = np.floor(mz64 / width).astype(np.int64)
    uniq, inv = np.unique(idx, return_inverse=True)
    centres = (uniq.astype(np.float64) + 0.5) * width
    if uniq.size == mz64.size:                        # nothing collided
        return centres, it64

    out = np.zeros(uniq.size, dtype=np.float64)
    if merge == "sqrt":
        np.add.at(out, inv, np.sqrt(np.clip(it64, 0.0, None)))
        out = out ** 2
    elif merge == "sum":
        np.add.at(out, inv, it64)
    else:
        np.maximum.at(out, inv, it64)
    return centres, out


def quantise_spectrum(mz, intensity, precursor_mz: float, width: float, *,
                      merge: str = "sqrt") -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply the rounding control to a whole spectrum, precursor included.

    This is the function to use. :func:`quantise_peaks` handles the fragment axis;
    this one also snaps the precursor to the same grid, which is load-bearing and
    easy to miss.

    Most peak encoders derive a neutral loss as ``precursor - mz``. If the peaks
    are quantised while the precursor keeps its full precision, that subtraction
    returns the precursor's mass defect to *every* token, and the arm that was
    supposed to carry only binned mass information silently regains a
    full-precision channel. The control then understates the value of the mass
    axis by an amount nothing in the training curves will reveal.

    Returns
    -------
    (mz, intensity, precursor_mz) : tuple
        Quantised peaks and the snapped precursor.

    Examples
    --------
    >>> mz = [67.0178, 67.0542, 121.0648]
    >>> inten = [100.0, 40.0, 25.0]
    >>> q_mz, q_int, q_prec = quantise_spectrum(mz, inten, 300.1234, width=0.5)
    >>> q_mz                        # the two isobars now share one peak
    array([ 67.25, 121.25])
    >>> round(q_prec, 2)
    300.25
    """
    q_mz, q_int = quantise_peaks(mz, intensity, width, merge=merge)
    if not width or width <= 0:
        return q_mz, q_int, float(precursor_mz)
    prec = (np.floor(np.float64(precursor_mz) / width) + 0.5) * width
    return q_mz, q_int, float(prec)


def absorbed_fraction(mz, width: float) -> float:
    """Fraction of peaks that fall into an already-occupied cell and are merged.

    The mechanism behind the control, measurable before any model is fitted, and
    worth reporting alongside a bin width: it says how much of the measurement the
    grid throws away. Report it on the same spectra the effects are measured on -
    held-out, if that is where the effects are - because the figure moves with the
    subset.

    Returns 0.0 for a non-positive width or a spectrum with no peaks.
    """
    mz64 = np.asarray(mz, dtype=np.float64).ravel()
    if not width or width <= 0 or mz64.size == 0:
        return 0.0
    idx = np.floor(mz64 / width).astype(np.int64)
    return float(1.0 - np.unique(idx).size / mz64.size)


class ConditioningParityError(AssertionError):
    """Raised when an arm does not respond to the side information it was given."""


def conditioning_response(predict: Callable[[Mapping], np.ndarray],
                          setting_a: Mapping, setting_b: Mapping) -> float:
    """How far an arm's output moves when only the side information changes.

    ``predict`` maps a mapping of conditioning variables to an output array, with
    everything else - the spectrum above all - held fixed between the two calls.
    Returns the maximum absolute difference. Zero means the arm cannot see those
    variables at all.

    Zero-initialised conditioning paths are a common trap. FiLM and adaptive
    layer-norm are often initialised so the modulation is the identity at step 0,
    which makes an untrained model return exactly 0.0 here whether or not it is
    wired correctly. Perturb the conditioning parameters off their initialisation
    before testing, or test a trained checkpoint; otherwise this measures the
    initialisation rather than the architecture.
    """
    a = np.asarray(predict(setting_a), dtype=np.float64)
    b = np.asarray(predict(setting_b), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"predict returned different shapes: {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def check_conditioning_parity(arms: Mapping[str, Callable[[Mapping], np.ndarray]],
                              setting_a: Mapping, setting_b: Mapping, *,
                              atol: float = 1e-6) -> dict:
    """Assert that every arm of an ablation responds to the conditioning variables.

    Run this before trusting any contrast between arms that are all nominally
    given the same side information. An arm whose response is zero is not a
    weaker model - it is a different experiment, and any increment measured
    against it includes the value of information the other arms had and it did
    not.

    Parameters
    ----------
    arms : mapping of name to predict-callable
        One entry per arm, each with the signature described in
        :func:`conditioning_response`.
    setting_a, setting_b : mapping
        Two conditioning settings that should produce different predictions. Make
        them genuinely different - a low and a high collision energy, two adducts,
        two instruments - since an arm can legitimately be near-invariant to a
        small change.
    atol : float
        Responses at or below this count as no response.

    Returns
    -------
    dict
        Arm name to response magnitude, for reporting.

    Raises
    ------
    ConditioningParityError
        If any arm's response is at or below ``atol``, naming every arm that
        failed.
    """
    responses = {name: conditioning_response(fn, setting_a, setting_b)
                 for name, fn in arms.items()}
    deaf = sorted(n for n, r in responses.items() if r <= atol)
    if deaf:
        detail = ", ".join(f"{n}={responses[n]:.3e}" for n in deaf)
        raise ConditioningParityError(
            f"{len(deaf)} of {len(responses)} arms do not respond to the "
            f"conditioning variables: {detail}. Any contrast against these arms "
            f"also prices the side information they cannot see. If the model is "
            f"untrained, check the conditioning path is not zero-initialised "
            f"before concluding it is unwired.")
    return responses
