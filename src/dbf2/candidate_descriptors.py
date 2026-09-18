"""Structural descriptors for every candidate structure (specification section 7).

Retrieval scores a candidate graph G by the posterior predictive density of its
descriptors Phi(G) under the model, so every structure that could be proposed needs
its BRICS presence vector in the *model's own* vocabulary. The training targets
cover only the molecules the model was fitted on; a candidate library is larger.

Presence bits are stored packed (512 bits = 64 bytes per structure), so the whole
273,934-structure library costs about 17 MB.

    python -m dbf2.candidate_descriptors --root . --library processed_v2/molecules.parquet --budget 150

Resumable: rerun the identical command until it prints DESCRIPTORS COMPLETE.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .brics import brics_pieces, MAX_HEAVY_ATOMS
from .config import Config

CHUNK = 20000


def _vocabulary(cfg: Config) -> List[str]:
    dic = pd.read_csv(cfg.paths.target_dict)
    return dic.loc[dic.family == "brics_presence", "pattern"].astype(str).tolist()


def run(cfg: Config, library: Path, budget: float = float("inf")) -> bool:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    out_dir = cfg.paths.prepared / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + budget

    vocab = _vocabulary(cfg)
    vidx = {p: i for i, p in enumerate(vocab)}
    lib = pd.read_parquet(library, columns=["inchikey14", "normalized_smiles",
                                            "molecular_formula", "mono_mass"])
    lib = lib.dropna(subset=["mono_mass"]).reset_index(drop=True)
    n = len(lib)
    n_chunks = (n + CHUNK - 1) // CHUNK

    for c in range(n_chunks):
        path = out_dir / f"bits_{c:03d}.npy"
        if path.exists():
            continue
        if time.time() > deadline:
            print(f"[descriptors] budget exhausted at chunk {c}/{n_chunks}; rerun to resume",
                  flush=True)
            return False
        sl = lib.normalized_smiles.iloc[c * CHUNK:(c + 1) * CHUNK]
        bits = np.zeros((len(sl), len(vocab)), dtype=np.uint8)
        skipped = 0
        for i, smi in enumerate(sl):
            m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            pieces, too_big = brics_pieces(m)
            skipped += int(too_big)
            for p in pieces:
                j = vidx.get(p)
                if j is not None:
                    bits[i, j] = 1
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, np.packbits(bits, axis=1))
        tmp.replace(path)
        print(f"[descriptors] chunk {c + 1}/{n_chunks} "
              f"({bits.sum(1).mean():.1f} pieces per structure, "
              f"{skipped} skipped above {MAX_HEAVY_ATOMS} atoms)", flush=True)

    packed = np.concatenate([np.load(out_dir / f"bits_{c:03d}.npy") for c in range(n_chunks)])
    np.save(out_dir / "presence_packed.npy", packed)
    lib.to_parquet(out_dir / "library.parquet")
    (out_dir / "meta.json").write_text(json.dumps(
        {"n_structures": int(n), "n_targets": len(vocab), "source": str(library),
         "packed_bytes": int(packed.shape[1])}, indent=1), encoding="utf-8")
    covered = float((np.unpackbits(packed, axis=1)[:, :len(vocab)].sum(1) > 0).mean())
    print(f"[descriptors] {n} structures, {len(vocab)} targets, "
          f"{covered:.1%} carry at least one vocabulary piece", flush=True)
    print("DESCRIPTORS COMPLETE", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--library", type=Path, required=True,
                    help="parquet with inchikey14, normalized_smiles, molecular_formula, mono_mass")
    ap.add_argument("--budget", type=float, default=float("inf"))
    a = ap.parse_args()
    cfg = Config(); cfg.paths.root = a.root
    run(cfg, a.library, a.budget)


if __name__ == "__main__":
    main()
