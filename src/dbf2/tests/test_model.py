"""Modules I and II: representation fidelity, pooling behaviour, collation."""
from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:                                   # pragma: no cover
    HAVE_TORCH = False

if HAVE_TORCH:
    from dbf2.config import ModelConfig
    from dbf2.encoder import FourierMZ
    from dbf2.measurement import accumulate_statistics
    from dbf2.model import DeepBayesFrag, NegBinHead
    from dbf2.dataset import collate


def _batch(B=5, V=4, K=40, J=64):
    mz, _ = (torch.rand(B, V, K, dtype=torch.float64) * 400 + 50).sort(-1)
    peak_mask = torch.ones(B, V, K, dtype=torch.bool)
    peak_mask[:, :, -5:] = False
    vm = torch.ones(B, V, dtype=torch.bool)
    if B > 1:
        vm[0, 2:] = False
    return {"mz": mz, "intensity": torch.rand(B, V, K), "peak_mask": peak_mask,
            "view_mask": vm, "precursor_mz": torch.full((B, V), 480.0, dtype=torch.float64),
            "adduct_id": torch.zeros(B, V, dtype=torch.long),
            "polarity_id": torch.zeros(B, V, dtype=torch.long),
            "instrument_id": torch.zeros(B, V, dtype=torch.long),
            "ce_ev": torch.rand(B, V) * 60,
            "ce_missing": torch.zeros(B, V, dtype=torch.long),
            "presence": (torch.rand(B, J) < 0.1).float(),
            "counts": torch.randint(0, 3, (B, 88)).float()}


def _cfg(encoder="peakset", pooling="hierarchical"):
    return ModelConfig(encoder=encoder, pooling=pooling, d_model=32, n_blocks=2,
                       n_heads=4, d_ff=64, fourier_scales=8, n_presence=64,
                       n_count=88, d_cov=16)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestFourier(unittest.TestCase):
    """The whole point of Module I is that the mass defect survives."""

    def setUp(self):
        self.f = FourierMZ(32, 2e-3, 1e3)

    def test_separates_isobaric_fragment_formulas(self):
        # C4H3O+ 67.0178 and C5H7+ 67.0542 fall in the same 0.5 Da bin
        a = torch.tensor([[67.0178], [67.0542]], dtype=torch.float64)
        self.assertGreater(float((self.f(a)[0] - self.f(a)[1]).abs().max()), 0.5)

    def test_one_millidalton_survives_the_float32_cast(self):
        """Phase at wavelength 2 mDa and m/z 342 exceeds 1e6; computing it in
        float32 would erase the defect entirely."""
        b = torch.tensor([[342.15610], [342.15710]], dtype=torch.float64)
        self.assertGreater(float((self.f(b)[0] - self.f(b)[1]).abs().max()), 1e-3)

    def test_deterministic_and_finite(self):
        c = torch.tensor([[342.1561], [342.1561]], dtype=torch.float64)
        self.assertEqual(float((self.f(c)[0] - self.f(c)[1]).abs().max()), 0.0)
        self.assertTrue(bool(torch.isfinite(
            self.f(torch.tensor([[1999.99999]], dtype=torch.float64))).all()))


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestModel(unittest.TestCase):
    def test_all_ablation_rungs_run(self):
        for encoder, pooling in (("binned", "mean"), ("peakset", "mean"),
                                 ("peakset", "hierarchical")):
            m = DeepBayesFrag(_cfg(encoder, pooling))
            b = _batch()
            out = m(b)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out["logit"], b["presence"])
            loss = loss + NegBinHead.nll(out["count_log_mu"],
                                         out["count_log_phi"], b["counts"]).mean()
            loss.backward()
            g = sum(p.grad.abs().sum().item() for p in m.parameters()
                    if p.grad is not None)
            self.assertTrue(bool(torch.isfinite(out["logit"]).all()))
            self.assertGreater(g, 0.0, f"{encoder}/{pooling} produced no gradient")

    def test_variance_decreases_with_views(self):
        """Specification equation (20): precision accumulates over views."""
        m = DeepBayesFrag(_cfg()).eval()
        b = _batch(B=1, V=6)
        seen = []
        with torch.no_grad():
            for v in range(1, 7):
                b2 = dict(b)
                b2["view_mask"] = torch.zeros(1, 6, dtype=torch.bool)
                b2["view_mask"][0, :v] = True
                seen.append(float(m(b2)["theta_var"].mean()))
        for i in range(len(seen) - 1):
            self.assertGreater(seen[i], seen[i + 1])

    def test_uncertainty_attenuates_probability(self):
        """Specification equation (22)."""
        m = DeepBayesFrag(_cfg()).eval()
        b = _batch(B=1, V=6)
        with torch.no_grad():
            o = m(b)
            att = float((torch.sigmoid(o["logit"]) - 0.5).abs().mean())
            raw = float((torch.sigmoid(o["theta_mean"]) - 0.5).abs().mean())
        self.assertLessEqual(att, raw + 1e-9)

    def test_statistics_are_additive_over_view_chunks(self):
        """This is what makes the specification's 'no cap on views' claim true."""
        m = DeepBayesFrag(_cfg()).eval()
        b = _batch(B=2, V=6)
        with torch.no_grad():
            h, cov = m.encode_views(b)
            vm = b["view_mask"]
            s1, s2 = accumulate_statistics(m.aggregator, h, cov, vm)
            s1a, s2a = accumulate_statistics(m.aggregator, h[:, :3], cov[:, :3], vm[:, :3])
            s1b, s2b = accumulate_statistics(m.aggregator, h[:, 3:], cov[:, 3:], vm[:, 3:])
        self.assertLess(float((s1 - (s1a + s1b)).abs().max()), 1e-4)
        self.assertLess(float((s2 - (s2a + s2b)).abs().max()), 1e-4)

    def test_masked_views_do_not_contribute(self):
        m = DeepBayesFrag(_cfg()).eval()
        b = _batch(B=1, V=4)
        with torch.no_grad():
            b["view_mask"] = torch.tensor([[True, True, False, False]])
            a = m(b)["theta_mean"].clone()
            b["mz"] = b["mz"].clone()
            b["mz"][0, 2:] = b["mz"][0, 2:] + 37.0      # perturb the masked views only
            c = m(b)["theta_mean"]
        self.assertLess(float((a - c).abs().max()), 1e-4)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class TestCollate(unittest.TestCase):
    def test_ragged_views_and_peaks(self):
        items = [
            {"group_id": 0, "mz": [np.array([100.1, 200.2]), np.array([50.5])],
             "intensity": [np.array([1.0, 0.5]), np.array([0.2])],
             "precursor_mz": np.array([300.0, 300.0]), "adduct_id": np.array([0, 0]),
             "polarity_id": np.array([0, 0]), "instrument_id": np.array([0, 0]),
             "ce_ev": np.array([20.0, 40.0]), "ce_missing": np.array([0, 0]),
             "n_views_total": 2, "presence": np.zeros(64, np.int8)},
            {"group_id": 1, "mz": [np.array([70.0, 80.0, 90.0, 120.0])],
             "intensity": [np.array([1.0, 0.3, 0.2, 0.1])],
             "precursor_mz": np.array([150.0]), "adduct_id": np.array([1]),
             "polarity_id": np.array([1]), "instrument_id": np.array([0]),
             "ce_ev": np.array([60.0]), "ce_missing": np.array([0]),
             "n_views_total": 1, "presence": np.ones(64, np.int8)}]
        b = collate(items)
        self.assertEqual(tuple(b["mz"].shape), (2, 2, 4))
        self.assertEqual(b["view_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(b["peak_mask"][0, 1].tolist(), [True, False, False, False])
        self.assertEqual(b["mz"].dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
