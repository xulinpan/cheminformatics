"""The research code and the published control package must agree.

``dbf2.dataset.quantise_merge`` is what the training pipeline actually calls;
``msms_controls.quantise_peaks`` is what the paper tells other people to use.
They are deliberately separate - the published package must not drag in torch or
this project's data layout - so nothing but a test stops them drifting apart.

Skipped when msms-controls is not installed, which is the normal state for
someone who only wants to reproduce the paper.
"""
import numpy as np
import pytest

from dbf2.dataset import quantise_merge

msms_controls = pytest.importorskip("msms_controls")

W = 0.5


@pytest.mark.parametrize("mode", ["sqrt", "sum", "max"])
def test_quantisation_matches_the_published_package(mode):
    rng = np.random.default_rng(0)
    mz = np.sort(rng.uniform(50.0, 900.0, 300))
    inten = rng.uniform(1.0, 1000.0, 300)
    a_mz, a_int = quantise_merge(mz, inten, W, mode=mode)
    b_mz, b_int = msms_controls.quantise_peaks(mz, inten, W, merge=mode)
    assert np.allclose(a_mz, b_mz), f"mass axes diverged for merge={mode}"
    assert np.allclose(a_int, b_int), f"intensities diverged for merge={mode}"


def test_collision_handling_matches_on_isobars():
    mz, inten = [67.0178, 67.0542], [100.0, 36.0]
    a_mz, a_int = quantise_merge(np.array(mz), np.array(inten), W)
    b_mz, b_int = msms_controls.quantise_peaks(mz, inten, W)
    assert np.allclose(a_mz, b_mz) and np.allclose(a_int, b_int)
