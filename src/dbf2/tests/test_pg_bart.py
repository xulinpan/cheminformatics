"""Stage 2 sampler: tree bookkeeping, horseshoe stability, signal recovery."""
from __future__ import annotations

import unittest

import numpy as np

from dbf2 import pg_bart as pb


class TestTree(unittest.TestCase):
    def test_pruned_children_are_not_reported_as_leaves(self):
        """Regression: pruned children remained in the node list and were later
        selected for a grow move, whose member lookup then raised KeyError."""
        rng = np.random.RandomState(0)
        n = 200
        X = rng.randn(n, 4)
        y = X[:, 0] + rng.randn(n) * 0.1
        ens = pb.SumOfTrees(n, n_trees=3, alpha=0.95, beta=2.0, leaf_sd=0.5, rng=rng)
        for _ in range(60):
            ens.step(X, y, np.ones(n))
            for tree in ens.trees:
                for leaf in tree.leaves():
                    self.assertIn(leaf, tree.members)
                    self.assertTrue(tree.alive[leaf])
                total = sum(len(v) for v in tree.members.values())
                self.assertEqual(total, n, "leaf membership must partition the data")


class TestSampler(unittest.TestCase):
    @staticmethod
    def _problem(seed=0, n=600, d=6, J=20, R=2):
        rng = np.random.RandomState(seed)
        H = rng.randn(n, d)
        B = np.stack([np.sin(2 * H[:, 0]) + 0.8 * (H[:, 1] > 0),
                      0.7 * H[:, 2] ** 2 - 0.7], axis=1)
        L = rng.randn(J, R) * 0.9
        offset = rng.randn(n, J) * 0.3
        true = B @ L.T
        theta = offset + true
        D = (rng.rand(n, J) < 1 / (1 + np.exp(-theta))).astype(float)
        return H, D, offset, true, theta, R

    def test_recovers_nonlinear_residual(self):
        H, D, offset, true, theta, R = self._problem()
        s = pb.PGBARTSampler(H, D, offset, None, None, n_factors=R, n_trees=8, seed=1)
        draws = s.run(600, 300, thin=5, verbose_every=0)
        est = draws["residual_mean"].mean(0)
        r = float(np.corrcoef(est.ravel(), true.ravel())[0, 1])
        self.assertGreater(r, 0.6, f"residual correlation {r:.3f} too low")
        self.assertLess(est.std(), 10.0, "latent scale diverged")

    def test_improves_on_the_deep_offset(self):
        H, D, offset, true, theta, R = self._problem()
        s = pb.PGBARTSampler(H, D, offset, None, None, n_factors=R, n_trees=8, seed=1)
        draws = s.run(600, 300, thin=5, verbose_every=0)
        Z = np.zeros_like(D)
        p_bart = pb.posterior_presence(draws, offset, Z, Z)
        p_off = 1 / (1 + np.exp(-offset))

        def nll(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(-(D * np.log(p) + (1 - D) * np.log(1 - p)).mean())
        self.assertLess(nll(p_bart), nll(p_off))

    def test_tau_stays_bounded_without_view_evidence(self):
        """Regression: with s2 = 0 the latent per-cell noise is unidentified and tau
        diverged, detaching theta from its mean and inflating every scale."""
        H, D, offset, true, theta, R = self._problem()
        s = pb.PGBARTSampler(H, D, offset, None, None, n_factors=R, n_trees=8, seed=3)
        draws = s.run(400, 200, thin=5, verbose_every=0)
        self.assertLessEqual(float(draws["tau"].max()), s.tau_hi + 1e-6)
        self.assertTrue(np.isfinite(draws["residual_mean"]).all())

    def test_pg_draws_match_known_moments(self):
        """PG(1, c) has mean tanh(c/2)/(2c), and 1/4 at c = 0."""
        rng = np.random.RandomState(0)
        for c in (0.0, 1.0, 3.0):
            x = pb.sample_pg(np.full(20000, c), rng)
            want = 0.25 if c == 0 else float(np.tanh(c / 2) / (2 * c))
            self.assertAlmostEqual(float(x.mean()), want, delta=0.02,
                                   msg=f"PG(1,{c}) mean")


if __name__ == "__main__":
    unittest.main()
