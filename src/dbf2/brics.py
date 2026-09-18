"""Shared BRICS decomposition with a size guard.

``BRICS.BRICSDecompose`` enumerates recursively and its cost grows sharply with the
number of cuttable bonds: on structures of a couple of hundred atoms a single call
can run for minutes. The competition's test precursors span 245 to 460 Da, roughly
35 heavy atoms, so structures far above that can never be mass-compatible
candidates and are skipped rather than waited for.

Training targets and candidate descriptors must come from the same function, or the
structural score compares vectors built two different ways.
"""
from __future__ import annotations

from typing import List, Tuple

MAX_HEAVY_ATOMS = 80        # about 1,000 Da, well above the test ceiling


def brics_pieces(mol, max_atoms: int = MAX_HEAVY_ATOMS) -> Tuple[List[str], bool]:
    """Return (attachment-labelled pieces, skipped_for_size)."""
    if mol is None:
        return [], False
    if mol.GetNumAtoms() > max_atoms:
        return [], True
    from rdkit.Chem import BRICS
    try:
        return list(BRICS.BRICSDecompose(mol, returnMols=False)), False
    except Exception:
        return [], False
