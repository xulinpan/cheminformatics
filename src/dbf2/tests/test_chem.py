"""Chemistry: adduct algebra, energy harmonisation, formula enumeration."""
from __future__ import annotations

import unittest

import numpy as np

from dbf2 import chem


class TestAdducts(unittest.TestCase):
    CASES = {"[M+H]+": (1, 1.00727646, 1), "[M-H]-": (1, -1.00727646, -1),
             "[M+Na]+": (1, 22.98922070, 1), "[M+NH4]+": (1, 18.03382555, 1),
             "[M+Cl]-": (1, 34.96940126, -1), "[M+CH2O2-H]-": (1, 44.99820285, -1),
             "[M-H2O+H]+": (1, -17.00328822, 1), "[2M+Na]+": (2, 22.98922070, 1),
             "[M+2H]2+": (1, 2.01455292, 2), "[M]+": (1, -0.00054858, 1)}

    def test_delta_and_charge(self):
        for name, (n, delta, z) in self.CASES.items():
            a = chem.parse_adduct(name)
            self.assertEqual(a.n_mer, n, name)
            self.assertEqual(abs(a.charge), abs(z), name)
            self.assertAlmostEqual(a.delta, delta, places=5, msg=name)

    def test_mz_round_trip(self):
        for name in self.CASES:
            a = chem.parse_adduct(name)
            m = 341.14879
            self.assertAlmostEqual(a.neutral_mass(a.mz(m)), m, places=8, msg=name)

    def test_known_precursor(self):
        # C17H19N5O3 as [M+H]+ is observed at 342.1561 in the competition test set
        m = chem.formula_mass("C17H19N5O3")
        self.assertAlmostEqual(chem.parse_adduct("[M+H]+").mz(m), 342.1561, places=3)

    def test_unparseable_returns_none(self):
        self.assertIsNone(chem.get_adduct("not-an-adduct"))


class TestFormula(unittest.TestCase):
    def test_parse_and_mass(self):
        self.assertEqual(chem.parse_formula("C17H19N5O3"),
                         {"C": 17, "H": 19, "N": 5, "O": 3})
        self.assertAlmostEqual(chem.formula_mass("C6H12O6"), 180.06339, places=4)

    def test_hill_notation(self):
        self.assertEqual(chem.format_formula({"C": 1, "H": 4}), "CH4")
        self.assertEqual(chem.format_formula({"C": 17, "H": 19, "N": 5, "O": 3}),
                         "C17H19N5O3")

    def test_rdbe(self):
        self.assertAlmostEqual(chem.rdbe(chem.parse_formula("C6H6")), 4.0)

    def test_enumeration_contains_truth(self):
        """The element caps must not exclude real structures.

        C11H16N6O at 249.1465 as [M+H]+ has six nitrogens on a 248 Da skeleton and
        was excluded by a mass-scaled nitrogen ceiling; this is the regression test.
        """
        for mz, adduct, truth in ((342.1561, "[M+H]+", "C17H19N5O3"),
                                  (249.1465, "[M+H]+", "C11H16N6O")):
            names = [f for f, _, _ in chem.enumerate_formulas(mz, adduct, ppm=10.0)]
            self.assertIn(truth, names, f"{truth} missing from enumeration")

    def test_enumeration_respects_tolerance(self):
        out = chem.enumerate_formulas(342.1561, "[M+H]+", ppm=5.0)
        self.assertTrue(all(abs(p) <= 5.0 + 1e-6 for _, _, p in out))

    def test_subformula_coverage_is_discriminative(self):
        """Peaks generated from a parent must cover better under that parent than
        under an unrelated formula of similar mass."""
        parent = chem.parse_formula("C17H19N5O3")
        sub = chem.valid_subformula_masses(parent, charge=1)
        self.assertIsNotNone(sub)
        rng = np.random.RandomState(0)
        mz = np.sort(rng.choice(sub[sub > 60], 40, replace=False))
        it = np.ones_like(mz)
        good = chem.subformula_coverage(mz, it, parent, ppm=5.0)
        bad = chem.subformula_coverage(mz, it, chem.parse_formula("C14H21F4N2O3"), ppm=5.0)
        self.assertGreater(good, 0.95)
        self.assertGreater(good, bad)


class TestEnergy(unittest.TestCase):
    def test_units(self):
        self.assertEqual(chem.harmonise_energy([20, 40, 60], "eV", 342.0), (40.0, 0))
        ev, miss = chem.harmonise_energy([30.0], "NCE", 500.0)
        self.assertAlmostEqual(ev, 30.0)
        self.assertEqual(miss, 0)

    def test_unknown_units_flagged_not_assumed(self):
        ev, miss = chem.harmonise_energy([20], "unknown", 342.0)
        self.assertTrue(np.isnan(ev))
        self.assertEqual(miss, 1)

    def test_instrument_family(self):
        self.assertEqual(chem.instrument_family("timsTOF"), "timsTOF")
        self.assertEqual(chem.instrument_family("LC-ESI-QFT"), "Orbitrap")
        self.assertEqual(chem.instrument_family("LC-ESI-QTOF"), "QTOF")
        self.assertEqual(chem.instrument_family(None), "unknown")


if __name__ == "__main__":
    unittest.main()
