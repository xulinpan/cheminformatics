"""Metrics, checked against cases with known answers."""
from __future__ import annotations

import unittest

import numpy as np

from dbf2 import evaluate as ev


class TestMetrics(unittest.TestCase):
    def test_average_precision_extremes(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(ev.average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        # worst ranking: positives land at ranks 3 and 4, so AP = (1/3 + 2/4) / 2
        self.assertAlmostEqual(ev.average_precision(y, np.array([0.9, 0.8, 0.2, 0.1])),
                               (1 / 3 + 2 / 4) / 2)
        self.assertTrue(np.isnan(ev.average_precision(np.zeros(4), np.random.rand(4))))

    def test_roc_auc_extremes_and_ties(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(ev.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(ev.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0)
        self.assertAlmostEqual(ev.roc_auc(y, np.full(4, 0.5)), 0.5)

    def test_ece_zero_when_perfectly_calibrated(self):
        y = np.array([0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(ev.expected_calibration_error(y, y), 0.0)

    def test_ece_detects_overconfidence(self):
        rng = np.random.RandomState(0)
        y = (rng.rand(5000) < 0.5).astype(float)
        good = np.full(5000, 0.5)
        bad = np.where(y > 0, 0.99, 0.01) * 0 + 0.99       # always confident, half wrong
        self.assertLess(ev.expected_calibration_error(y, good),
                        ev.expected_calibration_error(y, bad))

    def test_mrr(self):
        out = ev.mrr_at_k([["a", "b", "c"], ["x", "y"], ["q"]], ["b", "x", "zzz"])
        self.assertAlmostEqual(out["mrr_at_25"], (0.5 + 1.0 + 0.0) / 3)
        self.assertAlmostEqual(out["top_1"], 1 / 3)

    def test_stratified_calibration_bins(self):
        rng = np.random.RandomState(0)
        y = (rng.rand(400, 8) < 0.2).astype(float)
        p = rng.rand(400, 8)
        out = ev.stratified_calibration(p, y, {"views": rng.randint(1, 10, 400)})
        self.assertIn("1", out["views"])
        self.assertIn("9+", out["views"])
        for row in out["views"].values():
            self.assertGreaterEqual(row["n_molecules"], 5)

    def test_bootstrap_interval_brackets_the_mean(self):
        v = [0.5] * 60 + [0.1] * 40
        lo, hi = ev.bootstrap_ci(v)
        self.assertLess(lo, float(np.mean(v)))
        self.assertGreater(hi, float(np.mean(v)))


class TestRetrievalScores(unittest.TestCase):
    def test_bayes_score_prefers_the_matching_descriptor(self):
        from dbf2.retrieval import bayes_structural_score
        prob = np.array([0.9, 0.9, 0.1, 0.1])
        cands = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]], dtype=float)
        s = bayes_structural_score(prob, cands)
        self.assertEqual(int(np.argmax(s)), 0)

    def test_calibration_recovers_the_informative_term(self):
        """Given one informative and one noise term, the fitted weight on the noise
        term should be the smaller."""
        from dbf2.retrieval import GeneralizedPosterior
        rng = np.random.RandomState(0)
        queries, truth = [], []
        for _ in range(120):
            k = 20
            t = rng.randint(k)
            good = rng.randn(k); good[t] += 3.0
            queries.append({"bayes": good, "formula": rng.randn(k)})
            truth.append(t)
        gp = GeneralizedPosterior({"bayes": 0.5, "formula": 0.5})
        w = gp.calibrate(queries, truth, n_steps=200, lr=0.1)
        self.assertGreater(w["bayes"], w["formula"])


if __name__ == "__main__":
    unittest.main()
