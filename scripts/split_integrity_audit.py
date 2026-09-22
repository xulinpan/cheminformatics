"""Split-integrity audit: does any structure appear in more than one fold?

The fold assignment groups generic Murcko scaffolds, so the guarantee the paper
relies on is stronger than "no spectrum is reused": no *structure* may straddle
the boundary, and no *scaffold group* may either. This script checks both, plus
the identity rule underneath them, and prints the exact counts a Methods section
needs to state rather than assert.

    python scripts/split_integrity_audit.py --root . --dataset open
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PREP = {"default": "dbf2_prepared", "open": "dbf2_prepared_open",
        "msg": "dbf2_prepared_msg"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".")
    p.add_argument("--dataset", default="open", choices=sorted(PREP))
    a = p.parse_args()

    root = Path(a.root)
    prep = root / PREP[a.dataset]
    mol = pd.read_parquet(prep / "molecules.parquet")
    spec = pd.read_parquet(prep / "views" / "spectra.parquet")

    out = {"dataset": a.dataset, "n_structures": int(len(mol)),
           "n_spectra": int(len(spec))}

    # 1. molecular identity is the first 14 characters of the InChIKey: the
    #    skeleton block. It ignores stereochemistry, isotopic labelling and
    #    protonation, so stereoisomers, tautomers recorded as such, salts and
    #    charge states of one compound collapse to a single identifier and
    #    therefore to a single fold.
    out["identifier"] = "InChIKey first block (14 characters)"
    out["duplicate_identifiers_in_table"] = int(mol.inchikey14.duplicated().sum())

    # 2. no identifier in more than one fold
    folds_per_key = mol.groupby("inchikey14").fold.nunique()
    out["identifiers_in_multiple_folds"] = int((folds_per_key > 1).sum())

    # 3. every spectrum of a structure inherits that structure's fold
    # spectra.parquet already carries inchikey14; take only the fold from the
    # molecule table so the join cannot silently create suffixed duplicates
    j = spec.merge(mol[["mol_idx", "fold"]], on="mol_idx", how="left")
    out["spectra_without_a_fold"] = int(j.fold.isna().sum())
    out["spectra_whose_structure_spans_folds"] = int(
        j.groupby("inchikey14").fold.nunique().gt(1).sum())

    # 4. no scaffold group in more than one fold, which is the stronger claim
    key = np.where(mol.scaffold.to_numpy() == "", mol.inchikey14.to_numpy(),
                   mol.scaffold.to_numpy())
    g = pd.DataFrame({"key": key, "fold": mol.fold.to_numpy()})
    out["n_scaffold_groups"] = int(g.key.nunique())
    out["scaffold_groups_in_multiple_folds"] = int(
        g.groupby("key").fold.nunique().gt(1).sum())
    out["acyclic_structures_keyed_by_identifier"] = int((mol.scaffold == "").sum())

    out["fold_sizes"] = {str(k): int(v) for k, v in
                         mol.fold.value_counts().sort_index().items()}
    out["clean"] = (out["identifiers_in_multiple_folds"] == 0
                    and out["scaffold_groups_in_multiple_folds"] == 0
                    and out["spectra_whose_structure_spans_folds"] == 0
                    and out["spectra_without_a_fold"] == 0)

    for k, v in out.items():
        print(f"  {k}: {v}")
    print("\n  VERDICT:", "clean" if out["clean"] else "LEAKAGE DETECTED")

    dest = prep / "split_integrity_audit.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  wrote {dest}")


if __name__ == "__main__":
    main()
