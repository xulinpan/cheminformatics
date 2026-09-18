"""View-set datasets and collation.

A training item is one molecule together with a set of its spectra. Because the
pooling of equation (20) is a sum over views, subsampling views during training is
consistent with the model: it reduces the accumulated precision, which the
posterior then represents honestly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import Config
from .encoder import ADDUCT_IDX, POLARITY_IDX, INSTRUMENT_IDX, N_ADDUCT, N_POLARITY, N_INSTRUMENT


def _code(series: pd.Series, table: Dict[str, int], n: int) -> np.ndarray:
    return series.map(lambda x: table.get(x, n - 1)).to_numpy(np.int64)


class ViewStore:
    """Memory-mapped flat peak store plus the per-view metadata table."""

    def __init__(self, views_dir: Path, split: str = "train"):
        if split == "train":
            self.meta = pd.read_parquet(views_dir / "spectra.parquet")
            self.mz = np.load(views_dir / "peaks_mz.npy", mmap_mode="r")
            self.inten = np.load(views_dir / "peaks_int.npy", mmap_mode="r")
            self.group_key = self.meta.mol_idx.to_numpy(np.int64)
        else:
            self.meta = pd.read_parquet(views_dir / "test_spectra.parquet")
            self.mz = np.load(views_dir / "test_peaks_mz.npy", mmap_mode="r")
            self.inten = np.load(views_dir / "test_peaks_int.npy", mmap_mode="r")
            ids = sorted(self.meta.molecule_id.unique())
            self.id_of_index = ids
            lookup = {m: i for i, m in enumerate(ids)}
            self.group_key = self.meta.molecule_id.map(lookup).to_numpy(np.int64)
            order = np.argsort(self.group_key, kind="stable")
            self.meta = self.meta.iloc[order].reset_index(drop=True)
            self.group_key = self.group_key[order]

        self.offset = self.meta.offset.to_numpy(np.int64)
        self.n_peaks = self.meta.n_peaks.to_numpy(np.int64)
        self.precursor = self.meta.precursor_mz.to_numpy(np.float64)
        self.adduct_id = _code(self.meta.adduct, ADDUCT_IDX, N_ADDUCT)
        self.polarity_id = _code(self.meta.polarity, POLARITY_IDX, N_POLARITY)
        self.instrument_id = _code(self.meta.instrument_family, INSTRUMENT_IDX, N_INSTRUMENT)
        self.ce_ev = np.nan_to_num(self.meta.ce_ev.to_numpy(np.float32), nan=0.0)
        self.ce_missing = self.meta.ce_missing.to_numpy(np.int64)

        uniq, start = np.unique(self.group_key, return_index=True)
        end = np.append(start[1:], len(self.group_key))
        self.groups = uniq
        self.span = {int(g): (int(s), int(e)) for g, s, e in zip(uniq, start, end)}

    def peaks(self, row: int) -> Tuple[np.ndarray, np.ndarray]:
        o, n = self.offset[row], self.n_peaks[row]
        return np.asarray(self.mz[o:o + n]), np.asarray(self.inten[o:o + n])


def quantise_merge(mz: np.ndarray, inten: np.ndarray, width: float,
                   mode: str = "sqrt") -> Tuple[np.ndarray, np.ndarray]:
    """Snap peaks to a bin grid and merge the ones that collide.

    This is the rounding control of the representation ablation. Peaks are
    assigned to cells by ``floor(mz / width)``, exactly as ``BinnedEncoder``
    indexes them, and each surviving peak is placed at its cell centre. Peaks
    sharing a cell are merged; ``mode="sqrt"`` accumulates square-root intensity
    the way the binned vector does, so the merged peak carries the same total
    the binned model would have seen.

    Returns the peaks a full-precision encoder would receive if the mass axis
    had been discretised first. Nothing else about the spectrum changes.
    """
    if not width or width <= 0 or mz.size == 0:
        return mz, inten
    mz64 = np.asarray(mz, np.float64)
    idx = np.floor(mz64 / width).astype(np.int64)
    uniq, inv = np.unique(idx, return_inverse=True)
    centres = (uniq.astype(np.float64) + 0.5) * width
    if uniq.size == mz64.size:                       # nothing collided
        return centres, np.asarray(inten, np.float64)
    it64 = np.asarray(inten, np.float64)
    out = np.zeros(uniq.size, np.float64)
    if mode == "sqrt":
        np.add.at(out, inv, np.sqrt(np.clip(it64, 0.0, None)))
        out = out ** 2
    elif mode == "sum":
        np.add.at(out, inv, it64)
    elif mode == "max":
        np.maximum.at(out, inv, it64)
    else:
        raise ValueError(f"unknown merge mode {mode!r}; expected sqrt, sum or max")
    return centres, out


class MoleculeViewDataset(Dataset):
    """One item per molecule: a set of views plus the structural targets."""

    def __init__(self, store: ViewStore, group_ids: Sequence[int],
                 presence: Optional[np.ndarray] = None,
                 counts: Optional[np.ndarray] = None,
                 target_row: Optional[Dict[int, int]] = None,
                 max_views: int = 6, max_peaks: int = 256,
                 train: bool = True, seed: int = 2026,
                 fingerprints: Optional[np.ndarray] = None,
                 quantise_mz: float = 0.0, quantise_merge_mode: str = "sqrt"):
        self.store = store
        self.ids = [int(g) for g in group_ids if int(g) in store.span]
        self.presence, self.counts = presence, counts
        self.target_row = target_row
        self.max_views, self.max_peaks = max_views, max_peaks
        self.quantise_mz = float(quantise_mz or 0.0)
        self.quantise_merge_mode = quantise_merge_mode
        self.train = train
        self.fingerprints = fingerprints
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        gid = self.ids[i]
        s, e = self.store.span[gid]
        rows = np.arange(s, e)
        if self.max_views and len(rows) > self.max_views:
            rows = (self.rng.choice(rows, self.max_views, replace=False) if self.train
                    else rows[np.linspace(0, len(rows) - 1, self.max_views).astype(int)])
            rows.sort()

        mz_list, in_list = [], []
        for r in rows:
            mz, it = self.store.peaks(int(r))
            if self.max_peaks and mz.size > self.max_peaks:
                sel = np.argpartition(it, -self.max_peaks)[-self.max_peaks:]
                sel.sort()
                mz, it = mz[sel], it[sel]
            if self.quantise_mz > 0:
                # applied after the intensity cap, so the control and M0 see the
                # same peaks before discretisation
                mz, it = quantise_merge(mz, it, self.quantise_mz,
                                        self.quantise_merge_mode)
            mz_list.append(mz)
            in_list.append(it)

        item = {
            "group_id": gid,
            "mz": mz_list, "intensity": in_list,
            "precursor_mz": self.store.precursor[rows],
            "adduct_id": self.store.adduct_id[rows],
            "polarity_id": self.store.polarity_id[rows],
            "instrument_id": self.store.instrument_id[rows],
            "ce_ev": self.store.ce_ev[rows],
            "ce_missing": self.store.ce_missing[rows],
            "n_views_total": e - s,
        }
        if self.presence is not None and self.target_row is not None:
            tr = self.target_row.get(gid)
            if tr is not None:
                item["presence"] = self.presence[tr]
                if self.counts is not None:
                    item["counts"] = self.counts[tr]
                if self.fingerprints is not None:
                    item["fingerprint"] = self.fingerprints[tr]
        return item


def collate(items: List[dict]) -> Dict[str, torch.Tensor]:
    b = len(items)
    v_max = max(len(it["mz"]) for it in items)
    k_max = max((m.size for it in items for m in it["mz"]), default=1)
    k_max = max(k_max, 1)

    mz = np.zeros((b, v_max, k_max), np.float64)
    inten = np.zeros((b, v_max, k_max), np.float32)
    peak_mask = np.zeros((b, v_max, k_max), bool)
    view_mask = np.zeros((b, v_max), bool)
    prec = np.ones((b, v_max), np.float64)
    add = np.zeros((b, v_max), np.int64)
    pol = np.zeros((b, v_max), np.int64)
    ins = np.zeros((b, v_max), np.int64)
    ce = np.zeros((b, v_max), np.float32)
    cem = np.ones((b, v_max), np.int64)

    for i, it in enumerate(items):
        for j, (m, s) in enumerate(zip(it["mz"], it["intensity"])):
            k = m.size
            mz[i, j, :k] = m
            inten[i, j, :k] = s
            peak_mask[i, j, :k] = True
            view_mask[i, j] = True
        n = len(it["mz"])
        prec[i, :n] = it["precursor_mz"]
        add[i, :n] = it["adduct_id"]
        pol[i, :n] = it["polarity_id"]
        ins[i, :n] = it["instrument_id"]
        ce[i, :n] = it["ce_ev"]
        cem[i, :n] = it["ce_missing"]

    out = {
        "mz": torch.from_numpy(mz), "intensity": torch.from_numpy(inten),
        "peak_mask": torch.from_numpy(peak_mask), "view_mask": torch.from_numpy(view_mask),
        "precursor_mz": torch.from_numpy(prec), "adduct_id": torch.from_numpy(add),
        "polarity_id": torch.from_numpy(pol), "instrument_id": torch.from_numpy(ins),
        "ce_ev": torch.from_numpy(ce), "ce_missing": torch.from_numpy(cem),
        "group_id": torch.tensor([it["group_id"] for it in items], dtype=torch.long),
        "n_views": torch.tensor([len(it["mz"]) for it in items], dtype=torch.long),
        "n_views_total": torch.tensor([it["n_views_total"] for it in items], dtype=torch.long),
    }
    if "presence" in items[0]:
        out["presence"] = torch.from_numpy(np.stack([it["presence"] for it in items])).float()
    if "counts" in items[0]:
        out["counts"] = torch.from_numpy(np.stack([it["counts"] for it in items])).float()
    if "fingerprint" in items[0]:
        out["fingerprint"] = torch.from_numpy(np.stack([it["fingerprint"] for it in items]))
    return out


class PeakBucketBatchSampler(torch.utils.data.Sampler):
    """Batch molecules of similar total peak count together.

    Collation pads to the largest spectrum in the batch and attention cost grows
    with the square of that length, so one 256-peak spectrum makes every other
    spectrum in its batch as expensive. Shuffling within a pool and sorting inside
    it keeps batches homogeneous while preserving randomness across epochs.
    """

    def __init__(self, dataset: "MoleculeViewDataset", batch_size: int,
                 pool_multiplier: int = 50, shuffle: bool = True,
                 drop_last: bool = False, seed: int = 2026):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pool = max(batch_size * pool_multiplier, batch_size)
        self.shuffle, self.drop_last = shuffle, drop_last
        self.epoch = 0
        self.seed = seed
        store, mv = dataset.store, dataset.max_views
        cost = np.zeros(len(dataset.ids), dtype=np.int64)
        for i, gid in enumerate(dataset.ids):
            s, e = store.span[gid]
            n = store.n_peaks[s:e]
            if mv and n.size > mv:
                n = np.sort(n)[-mv:]
            cost[i] = int(n.max()) if n.size else 0
        self.cost = cost

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self.cost)) if self.shuffle else np.arange(len(self.cost))
        batches = []
        for start in range(0, len(order), self.pool):
            chunk = order[start:start + self.pool]
            chunk = chunk[np.argsort(self.cost[chunk], kind="stable")]
            for i in range(0, len(chunk), self.batch_size):
                b = chunk[i:i + self.batch_size]
                if self.drop_last and len(b) < self.batch_size:
                    continue
                batches.append(b.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        n = len(self.cost)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


def load_targets(cfg: Config, keep_only_above_prevalence: bool = False):
    """Return (presence, counts, row index by mol_idx, presence names, count names)."""
    tgt = pd.read_parquet(cfg.paths.targets)
    dic = pd.read_csv(cfg.paths.target_dict)
    pres_names = dic.loc[dic.family == "brics_presence", "variable"].tolist()
    cnt_names = dic.loc[dic.family == "count", "variable"].tolist()
    if keep_only_above_prevalence:
        keep = set(dic.loc[dic.keep_for_bayes, "variable"])
        pres_names = [c for c in pres_names if c in keep]
    presence = tgt[pres_names].to_numpy(np.int8)
    counts = tgt[cnt_names].to_numpy(np.int16)
    row_of = {int(m): i for i, m in enumerate(tgt.mol_idx.to_numpy())}
    return presence, counts, row_of, pres_names, cnt_names


def split_molecules(cfg: Config, exclude_oracle: bool = True):
    """Fold assignment with the leak-derived oracle structures held out of training."""
    mol = pd.read_parquet(cfg.paths.molecules)
    if exclude_oracle and "is_oracle" in mol.columns:
        oracle = mol.loc[mol.is_oracle, "mol_idx"].to_numpy()
        pool = mol.loc[~mol.is_oracle]
    else:
        oracle = np.array([], dtype=np.int64)
        pool = mol
    tr = pool.loc[pool.fold.isin(cfg.train.train_folds), "mol_idx"].to_numpy()
    va = pool.loc[pool.fold == cfg.train.val_fold, "mol_idx"].to_numpy()
    te = pool.loc[pool.fold == cfg.train.test_fold, "mol_idx"].to_numpy()
    return {"train": tr, "val": va, "test": te, "oracle": oracle}
