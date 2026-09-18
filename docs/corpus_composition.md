# Corpus composition and the reproducibility route

The manuscript's experiments run on the CASMI 2026 training corpus, which the
competition organisers distribute through Kaggle. *Journal of Cheminformatics*
publishes only work "entirely reproducible by third parties," with data reachable
"without the need for registration, login or agreement with license terms other
than Creative Commons licenses." A Kaggle download requires an account, a login and
acceptance of competition rules, so the corpus as distributed does not satisfy that
policy. This note records what the corpus is actually made of and what that implies.

## What is in the file

`train.parquet` is 3.03 GB, 21 row groups, **2,539,608 spectra** over **275,810
distinct structures** (InChIKey14), 18 columns. The paper uses the first two row
groups — 262,144 spectra, 10.3% of the file. `test.parquet` holds 1,213 spectra
over 400 molecules.

The `ingest_lib` column names each spectrum's source library:

| source library | spectra | share | structures | redistribution |
|----------------|--------:|------:|-----------:|----------------|
| enveda-180 | 1,153,785 | 45.4% | 182,941 | in-house, not redistributable |
| pluskal_ms2 | 527,581 | 20.8% | 46,821 | likely public — verify |
| riken | 347,171 | 13.7% | 15,892 | likely public — verify |
| gnps | 220,849 | 8.7% | 45,750 | open (CC) |
| massbank | 101,727 | 4.0% | 9,180 | open (CC) |
| mona | 92,416 | 3.6% | 11,681 | open, but licensed per record |
| spectraverse | 50,933 | 2.0% | 9,631 | unverified |
| msdial | 40,765 | 1.6% | 9,127 | likely public — verify |
| drug_plus | 2,545 | 0.1% | 2,539 | unverified |
| enveda-np-examples | 1,184 | <0.1% | 250 | in-house, not redistributable |
| masaryk | 652 | <0.1% | 416 | likely public — verify |

## The consequence

Just under half the corpus (45.4%) is Enveda's in-house library and cannot be
redistributed under any arrangement we control. The rest derives from public
libraries.

The three unambiguously open sources — GNPS, MassBank and MoNA — together hold
**414,992 spectra over 52,428 structures**. That is **1.58× the data the paper
currently uses**. Rebuilding the four-arm ladder on openly licensed data therefore
does not mean scaling the experiment down; it means scaling it up from where it
stands today, on data a reader can obtain without an account.

Adding the four probably-public sources (Pluskal, RIKEN, MS-DIAL, Masaryk) would
give 1,384,639 spectra over 95,157 structures, 5.28× the current scale, but each
one's licence needs checking before it can be relied on.

Coverage in the open subset is adequate: 245,453 `[M+H]+`, 84,808 `[M-H]-`, 29,837
`[M+Na]+`, and a spread of instruments (119,782 Orbitrap, 69,235 LC-ESI-QTOF,
48,719 ESI-QFT, 36,698 LC-ESI-QFT, 32,420 qTof).

## What this does not license

The open subset is identified *inside* a Kaggle-distributed file. Redistributing
Kaggle's copy of those rows is not the same as obtaining them from their sources.
The route that satisfies the policy is to pull the records from GNPS, MassBank and
MoNA directly, filter MoNA by per-record licence, rebuild the splits and targets,
rerun the ladder, and deposit the assembled corpus with a DOI. The table above is
useful because it says which sources to pull and roughly how much each contributes,
not because it makes the existing file redistributable.

## Scale is not a data problem

All 21 row groups are present locally, so the manuscript's "two of twenty-one row
groups" limitation is a compute decision, not a data-availability one. A four-arm
ablation at the current scale takes about 31 minutes end to end (M0 3.1, M1R 9.0,
M1 8.9, M2 10.4). At full corpus scale that is roughly 5 hours per ablation and
about 16 hours for three seeds, before preprocessing — an overnight job rather than
a project.

## The subset as built

`scripts/make_open_subset.py` writes `data/train_open.parquet` from the three open
sources, preserving the schema so every downstream stage runs unchanged:

| | row groups 0-1 (used so far) | open subset |
|---|---:|---:|
| spectra | 262,144 | 414,992 |
| structures | 43,628 | 52,428 |
| peaks | — | 78,025,942 |
| source libraries | 2 (99.0% enveda-180) | 3 (GNPS, MassBank, MoNA) |
| distinct instruments | 2 | 66 |
| distinct adducts | 16 | 112 |
| dominant instrument | timsTOF, 99% | Orbitrap 29%, LC-ESI-QTOF 17% |
| held-out structures present | 400 of 400 | 2 of 400 |

The last row matters for the leak: the original corpus contained every one of the
400 competition structures, which is how the leak arose and why all 400 had to be
excluded from training. The open subset contains two of them, so the exclusion
remains necessary but is no longer material.

The file is selected by name rather than by editing code:

```bash
python -m dbf2 prepare --root . --dataset open
python -m dbf2 ablate  --root . --dataset open --rungs M0 M1R M1 M2 --epochs 15
python -m dbf2 ablate  --root . --dataset open --rungs M0 M1R M1 M2 --epochs 15 --seed 7
```

`--dataset open` reads `data/train_open.parquet` and writes to
`dbf2_prepared_open/` and `dbf2_runs_open/`, so the existing single-library
artefacts are left untouched and the two can be compared.
