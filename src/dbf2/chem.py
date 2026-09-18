"""Chemistry utilities: adduct algebra, collision-energy harmonisation, formula enumeration.

Specification sections 2.2 (energy harmonisation) and 6 (molecular formula).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------ atomic masses
ELECTRON = 0.000548579909065
MONO: Dict[str, float] = {
    "C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.9949146196,
    "P": 30.97376163, "S": 31.97207100, "Cl": 34.96885268, "Br": 78.9183371,
    "F": 18.99840322, "I": 126.904473, "Na": 22.98976928, "K": 38.96370649,
    "Si": 27.9769265325, "Se": 79.9165213, "B": 11.0093054,
}
VALENCE: Dict[str, float] = {
    "C": 4, "H": 1, "N": 3, "O": 2, "P": 3, "S": 2, "Cl": 1, "Br": 1,
    "F": 1, "I": 1, "Na": 1, "K": 1, "Si": 4, "Se": 2, "B": 3,
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_ADDUCT = re.compile(r"^\[(\d*)M(.*)\](\d*)([+-])$")
_ADDUCT_PART = re.compile(r"([+-])(\d*)([A-Za-z0-9]+)")


def parse_formula(formula: str) -> Dict[str, int]:
    """'C17H19N5O3' -> {'C': 17, 'H': 19, 'N': 5, 'O': 3}."""
    out: Dict[str, int] = {}
    pos = 0
    for m in _FORMULA_TOKEN.finditer(formula):
        if m.start() != pos:
            raise ValueError(f"cannot parse formula {formula!r} at position {pos}")
        pos = m.end()
        el, n = m.group(1), m.group(2)
        if el not in MONO:
            raise ValueError(f"unknown element {el!r} in {formula!r}")
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    if pos != len(formula):
        raise ValueError(f"trailing characters in formula {formula!r}")
    return out


def formula_mass(formula: str | Dict[str, int]) -> float:
    d = parse_formula(formula) if isinstance(formula, str) else formula
    return sum(MONO[el] * n for el, n in d.items())


def rdbe(counts: Dict[str, int]) -> float:
    """Ring plus double-bond equivalent; a structural plausibility filter."""
    return 1.0 + 0.5 * sum(n * (VALENCE.get(el, 2) - 2) for el, n in counts.items())


@dataclass(frozen=True)
class Adduct:
    name: str
    n_mer: int
    delta: float      # additive mass shift, electrons included
    charge: int       # signed

    def mz(self, neutral_mass: float) -> float:
        return (self.n_mer * neutral_mass + self.delta) / abs(self.charge)

    def neutral_mass(self, mz: float) -> float:
        return (mz * abs(self.charge) - self.delta) / self.n_mer


def parse_adduct(name: str) -> Adduct:
    """Parse '[2M+CH2O2-H]-' and similar into an Adduct.

    m/z = (n * M_neutral + delta) / |z|,  delta = sum(+-formula) - z * m_e
    """
    m = _ADDUCT.match(name.strip())
    if not m:
        raise ValueError(f"unparseable adduct {name!r}")
    n_mer = int(m.group(1)) if m.group(1) else 1
    body, zdigits, zsign = m.group(2), m.group(3), m.group(4)
    charge = (int(zdigits) if zdigits else 1) * (1 if zsign == "+" else -1)

    delta = 0.0
    pos = 0
    for part in _ADDUCT_PART.finditer(body):
        if part.start() != pos:
            raise ValueError(f"unparseable adduct body {body!r} in {name!r}")
        pos = part.end()
        sign = 1.0 if part.group(1) == "+" else -1.0
        mult = int(part.group(2)) if part.group(2) else 1
        delta += sign * mult * formula_mass(part.group(3))
    if pos != len(body):
        raise ValueError(f"trailing characters in adduct body {body!r}")
    delta -= charge * ELECTRON
    return Adduct(name=name, n_mer=n_mer, delta=delta, charge=charge)


_ADDUCT_CACHE: Dict[str, Optional[Adduct]] = {}


def get_adduct(name: str) -> Optional[Adduct]:
    """Cached, non-raising adduct lookup. Returns None for unparseable names."""
    if name not in _ADDUCT_CACHE:
        try:
            _ADDUCT_CACHE[name] = parse_adduct(name)
        except (ValueError, KeyError):
            _ADDUCT_CACHE[name] = None
    return _ADDUCT_CACHE[name]


# -------------------------------------------------------- collision energy, eq (5)
def harmonise_energy(values, units: Optional[str], precursor_mz: float) -> Tuple[float, int]:
    """Return (mean energy in eV, missing indicator).

    NCE is converted with the singly charged Thermo convention eV = NCE * m/z / 500.
    Unknown units yield (nan, 1) rather than being silently placed on the eV scale.
    """
    if values is None:
        return float("nan"), 1
    v = np.abs(np.asarray(values, dtype=np.float64).ravel())
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), 1
    if units in ("eV", "V", "ev", "v"):
        return float(v.mean()), 0
    if units == "NCE":
        return float(v.mean() * precursor_mz / 500.0), 0
    return float("nan"), 1


INSTRUMENT_PATTERNS = (
    ("timsTOF", ("timstof",)),
    ("Orbitrap", ("orbitrap", "qft", "itft", "hybrid ft", "q-exactive", "exactive",
                  "fusion", "lumos", "astral")),
    ("QTOF", ("tof",)),
)


def instrument_family(name: Optional[str]) -> str:
    if not name:
        return "unknown"
    low = str(name).lower()
    for fam, keys in INSTRUMENT_PATTERNS:
        if any(k in low for k in keys):
            return fam
    return "other"


# ----------------------------------------------- formula enumeration, eq (33)-(34)
def enumerate_formulas(
    precursor_mz: float,
    adduct_name: str,
    ppm: float = 10.0,
    elements: Sequence[str] = ("C", "H", "N", "O", "P", "S", "Cl", "Br", "F"),
    rdbe_range: Tuple[float, float] = (-0.5, 40.0),
    max_results: int = 5000,
) -> List[Tuple[str, float, float]]:
    """Enumerate neutral formulas consistent with a precursor m/z and adduct.

    Returns a list of (formula string, neutral mass, ppm error), sorted by |ppm|.
    Bounds follow the standard seven-golden-rules style heuristics.
    """
    add = get_adduct(adduct_name)
    if add is None or not np.isfinite(precursor_mz) or precursor_mz <= 0:
        return []
    target = add.neutral_mass(float(precursor_mz))
    if target <= 0:
        return []
    tol = abs(target) * ppm * 1e-6 * max(add.n_mer, 1)

    els = [e for e in elements if e in MONO]
    if "C" not in els or "H" not in els:
        raise ValueError("enumeration requires at least C and H")
    hetero = [e for e in els if e not in ("C", "H")]

    # element-count ceilings from the target mass
    cmax = int(target // MONO["C"]) + 1
    caps = {e: min(int(target // MONO[e]), _heuristic_cap(e, target)) for e in hetero}

    results: List[Tuple[str, float, float]] = []

    def recurse(idx: int, counts: Dict[str, int], mass: float) -> None:
        if len(results) >= max_results:
            return
        if idx == len(hetero):
            rem = target - mass
            if rem < -tol:
                return
            # choose C and H to close the remaining mass
            for nc in range(cmax + 1):
                mc = mass + nc * MONO["C"]
                if mc > target + tol:
                    break
                nh_f = (target - mc) / MONO["H"]
                for nh in (int(np.floor(nh_f)), int(np.ceil(nh_f))):
                    if nh < 0:
                        continue
                    tot = mc + nh * MONO["H"]
                    err = tot - target
                    if abs(err) > tol:
                        continue
                    cand = dict(counts)
                    cand["C"], cand["H"] = nc, nh
                    cand = {k: v for k, v in cand.items() if v > 0}
                    if not cand:
                        continue
                    r = rdbe(cand)
                    if not (rdbe_range[0] <= r <= rdbe_range[1]):
                        continue
                    if r != int(r) and abs(r - round(r)) not in (0.0, 0.5):
                        continue
                    if nh > 2 * nc + counts.get("N", 0) + 2 + counts.get("P", 0):
                        continue      # valence ceiling on hydrogen
                    results.append((format_formula(cand), tot,
                                    err / target * 1e6 if target else 0.0))
                    if len(results) >= max_results:
                        return
            return
        el = hetero[idx]
        for n in range(caps[el] + 1):
            m2 = mass + n * MONO[el]
            if m2 > target + tol:
                break
            counts[el] = n
            recurse(idx + 1, counts, m2)
        counts.pop(el, None)

    recurse(0, {}, 0.0)
    results.sort(key=lambda t: abs(t[2]))
    return results


# Absolute element ceilings in the small-molecule regime, in the spirit of the
# seven golden rules. A ratio scaled by mass alone is too tight below ~400 Da: at
# 248 Da a 20-per-1000 rule caps nitrogen at 5 and so excludes C11H16N6O, a real
# test structure. The caps below are absolute for mass <= 500 and grow above it.
_ABSOLUTE_CAP = {"N": 20, "O": 25, "P": 9, "S": 12, "Cl": 12, "Br": 8, "F": 20,
                 "I": 6, "Si": 8, "Se": 4, "B": 4, "Na": 2, "K": 2}


def _heuristic_cap(element: str, mass: float) -> int:
    """Upper bound on a heteroatom count, from both the mass and a chemistry ceiling."""
    base = _ABSOLUTE_CAP.get(element, 10)
    if mass > 500.0:
        base = int(base * mass / 500.0)
    by_mass = int(mass // MONO[element])
    return max(1, min(base, by_mass))


def format_formula(counts: Dict[str, int]) -> str:
    """Hill notation."""
    parts = []
    if counts.get("C"):
        parts.append("C" + (str(counts["C"]) if counts["C"] > 1 else ""))
        if counts.get("H"):
            parts.append("H" + (str(counts["H"]) if counts["H"] > 1 else ""))
        rest = sorted(k for k in counts if k not in ("C", "H") and counts[k])
    else:
        rest = sorted(k for k in counts if counts[k])
    for el in rest:
        parts.append(el + (str(counts[el]) if counts[el] > 1 else ""))
    return "".join(parts)


def valid_subformula_masses(parent: Dict[str, int], charge: int = 1,
                            max_space: int = 4_000_000) -> Optional[np.ndarray]:
    """Sorted ion masses of the chemically admissible subformulas of ``parent``.

    Unconstrained subformula enumeration is far too permissive: almost every peak
    then matches something and the coverage feature carries no information. Two
    standard constraints are imposed, which is what makes the feature discriminate:

      * ring-plus-double-bond equivalent in [-0.5, parent RDBE];
      * the hydrogen ceiling h <= 2c + n + p + 2.

    Returns None when the combinatorial space exceeds ``max_space``.
    """
    els = [e for e in parent if parent[e] > 0]
    if not els:
        return None
    caps = [parent[e] for e in els]
    size = 1
    for c in caps:
        size *= (c + 1)
        if size > max_space:
            return None
    mesh = np.array(np.meshgrid(*[np.arange(c + 1) for c in caps],
                                indexing="ij")).reshape(len(els), -1)
    masses = np.array([MONO[e] for e in els])
    val = np.array([VALENCE.get(e, 2) - 2 for e in els], dtype=np.float64)
    r = 1.0 + 0.5 * (val @ mesh)
    keep = (r >= -0.5) & (r <= rdbe(parent) + 1e-9)
    idx = {e: i for i, e in enumerate(els)}
    if "H" in idx:
        h = mesh[idx["H"]]
        ceiling = (2 * mesh[idx["C"]] if "C" in idx else 0) \
            + (mesh[idx["N"]] if "N" in idx else 0) \
            + (mesh[idx["P"]] if "P" in idx else 0) + 2
        keep &= h <= ceiling
    keep &= mesh.sum(axis=0) > 0
    if not keep.any():
        return None
    shift = -ELECTRON if charge > 0 else ELECTRON
    return np.sort(masses @ mesh[:, keep] + shift)


def subformula_coverage(
    peak_mz: np.ndarray,
    peak_intensity: np.ndarray,
    parent: Dict[str, int],
    ppm: float = 10.0,
    charge: int = 1,
) -> float:
    """Fraction of fragment intensity assignable to an admissible subformula of
    ``parent``. The coverage feature of specification equation (34).

    This is a bounded compositional check, not a fragmentation tree: it asks whether
    a peak's mass is reachable, not whether a connected fragment exists.
    """
    if peak_mz.size == 0:
        return 0.0
    sub = valid_subformula_masses(parent, charge)
    if sub is None:
        return float("nan")
    tol = peak_mz * ppm * 1e-6
    idx = np.searchsorted(sub, peak_mz)
    ok = np.zeros(peak_mz.shape, dtype=bool)
    for off in (-1, 0):
        j = np.clip(idx + off, 0, sub.size - 1)
        ok |= np.abs(sub[j] - peak_mz) <= tol
    total = peak_intensity.sum()
    return float(peak_intensity[ok].sum() / total) if total > 0 else 0.0
