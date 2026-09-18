"""Extract the openly licensed subset of the CASMI 2026 training corpus.

The corpus file is ordered by source library, so a contiguous slice of row groups
selects a library rather than sampling the corpus. This writes a subset holding
only the spectra contributed by GNPS, MassBank and MoNA -- the three sources whose
records are openly licensed -- preserving the schema so every downstream stage runs
unchanged.

The subset is NOT itself redistributable: these rows are identified inside a
Kaggle-distributed file. It exists so the experiments run on data a third party
could obtain independently from GNPS, MassBank and MoNA. See
docs/corpus_composition.md.

Usage:  python scripts/make_open_subset.py [--root .] [--out data/train_open.parquet]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

OPEN_LIBRARIES = ("gnps", "massbank", "mona")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--libraries", nargs="+", default=list(OPEN_LIBRARIES))
    a = ap.parse_args()

    src = a.root / "data" / "train.parquet"
    out = a.out or (a.root / "data" / "train_open.parquet")
    keep = pa.array(list(a.libraries))

    f = pq.ParquetFile(src)
    n_groups = f.metadata.num_row_groups
    print(f"source     : {src}  ({f.metadata.num_rows:,} rows, {n_groups} row groups)")
    print(f"keeping    : {', '.join(a.libraries)}")
    print(f"destination: {out}", flush=True)

    writer = None
    kept = seen = 0
    t0 = time.time()
    try:
        for i in range(n_groups):
            tbl = f.read_row_group(i)
            seen += tbl.num_rows
            mask = pc.is_in(tbl.column("ingest_lib"), value_set=keep)
            sub = tbl.filter(mask)
            if sub.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(out, tbl.schema, compression="snappy")
                writer.write_table(sub)
                kept += sub.num_rows
            print(f"  row group {i:2d}/{n_groups - 1}  read {tbl.num_rows:>7,}  "
                  f"kept {sub.num_rows:>7,}  running total {kept:>9,}  "
                  f"[{time.time() - t0:6.1f}s]", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        sys.exit("no rows matched; nothing written")
    size = out.stat().st_size
    print(f"\nread {seen:,} rows, wrote {kept:,} ({kept / seen:.1%})")
    print(f"output {size / 1e6:.0f} MB in {time.time() - t0:.0f}s")

    chk = pq.ParquetFile(out)
    print(f"verify: {chk.metadata.num_rows:,} rows, {chk.metadata.num_row_groups} row groups, "
          f"{chk.metadata.num_columns} columns")
    assert chk.metadata.num_rows == kept, "row count mismatch after write"
    assert chk.schema_arrow.equals(f.schema_arrow), "schema drifted"
    print("schema identical to source; row count matches")


if __name__ == "__main__":
    main()
