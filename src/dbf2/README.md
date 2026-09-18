# DeepBayesFrag-MS v2 — implementation

Python implementation of `DeepBayesFrag_MS_v2.tex`. Section numbers below refer to
that document.

The central object is a **hierarchical measurement model over a molecule's set of
spectra**: each view contributes evidence with a learned, acquisition-dependent and
target-specific precision, so the substructure posterior knows how much and what
kind of evidence it had.

```
l_ivj | theta_ij, C_iv ~ N( alpha_j(c) + beta_j(c) theta_ij , sigma_j(c)^2 )

v_ij^-1 = tau_ij^-2 + sum_v w_ivj ,     w_ivj = beta_j(c)^2 / sigma_j(c)^2
m_ij    = v_ij ( mu_ij tau_ij^-2 + sum_v w_ivj ( l_ivj - alpha_j(c) ) / beta_j(c) )
P(D_ij = 1 | X) = sigmoid( m_ij / sqrt(1 + pi v_ij / 8) )
```

## Layout

| file | specification | contents |
|---|---|---|
| `config.py` | — | every tunable value; `Config.ablation("M2", root)` builds a rung |
| `chem.py` | 2.2, 6 | adduct algebra, collision-energy harmonisation, formula enumeration, subformula coverage |
| `prepare.py` | 13 (Stage 0) | cleaning, flat peak store, folds, targets, oracle key |
| `dataset.py` | — | view-set dataset and ragged collation |
| `encoder.py` | 3 | Fourier peak tokens, FiLM set Transformer, binned encoder for M0 |
| `measurement.py` | **4** | the measurement model and conjugate pooling |
| `model.py` | 3–5 | assembly, negative-binomial count head, contrastive projection |
| `train.py` | 13 (Stage 1) | training loop, early stopping, reports |
| `pg_bart.py` | 5.3 | Polya-Gamma augmentation and low-rank BART |
| `bayes_stage.py` | 13 (Stage 2) | extracts statistics, runs the sampler, compares M3 with M2 |
| `candidates.py` | 6, 7 | formula scoring, candidate construction, recall |
| `retrieval.py` | 9, 12.1 | generalized posterior, weight calibration, MRR@25 |
| `evaluate.py` | 14 | AUPRC, ECE, **stratified calibration**, MRR |
| `cli.py` | — | `prepare`, `oracle`, `train`, `bayes`, `ablate` |

## Install

```powershell
cd D:\research2026\kaggle
.\.venv\Scripts\python.exe -m pip install polyagamma
```

Everything else (torch, numpy, pandas, pyarrow, rdkit) is already in `.venv`.
`polyagamma` is optional but strongly recommended: without it `pg_bart` falls back
to a truncated-series sampler that is correct only up to a 1/K truncation bias and
is markedly slower. `python -m dbf2 bayes` prints which one it used.

**Always run from `D:\research2026\kaggle`**, the directory that contains `dbf2\`.
The commands below assume that.

## Run

### 0. Smoke test, about ten minutes

Confirms the whole chain works before committing hours to it.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s dbf2\tests -v
.\.venv\Scripts\python.exe -m dbf2 prepare --root . --max-row-groups 2
.\.venv\Scripts\python.exe -m dbf2 train   --root . --rung M2 --epochs 2 --limit-molecules 1500
```

Then delete `dbf2_prepared\` before the real run: a partial preparation is recorded
in `manifest.json` and would otherwise be treated as complete.

### 1. Stage 0, preparation

```powershell
.\.venv\Scripts\python.exe -m dbf2 prepare --root . --budget 1800
```

Resumable: rerun the identical command until it prints `STAGE 0 COMPLETE`. Expect
roughly 4.5 GB in `dbf2_prepared\views\` plus the same again in `part_*_mz.raw`
intermediates, which can be deleted by hand once `manifest.json` records
`"regroup"`. Writes `oracle_test_key.csv`; see the leak section below.

### 2. Stage 1, the ablation ladder

`M2` against `M1` is the primary claim; `M1` against `M0` tests the representation
change alone.

```powershell
.\.venv\Scripts\python.exe -m dbf2 ablate --root . --rungs M0 M1 M2 --epochs 15
```

Results land in `dbf2_runs\<rung>\<rung>_report.json`, with
`dbf2_runs\ablation_summary.json` on top. The number that decides the claim is
`stratified_calibration`, not the aggregate: the hierarchical model should win
most in the `views_used` = 1 and 2 strata.

### 3. Stage 2, the Bayesian layer

```powershell
.\.venv\Scripts\python.exe -m dbf2 bayes --root . --run dbf2_runs\M2 --iter 1500 --burn 750
```

Prints `M2_deep_only` beside `M3_with_bart` on the same held-out molecules, which
is the M3-versus-M2 comparison.

## Memory and throughput

Attention cost grows with the square of the longest spectrum in a batch and
linearly in `batch x views`, so those three settings interact. Peak host memory
above the torch baseline, measured with `d_model=128`, four blocks, 512 targets:

| batch | views | peaks | memory |
|---|---|---|---|
| 16 | 4 | 128 | 0.9 GB |
| 32 | 4 | 128 | 1.8 GB |
| 32 | 4 | 256 | 5.1 GB |

Defaults are `batch_molecules=16`, `views_per_molecule=4`, `max_peaks=128`, which
keeps every peak for 89 percent of spectra (median is 38, p90 is 135). Batches are
bucketed by peak count so one long spectrum does not inflate the whole batch. On an
RTX 3070 raise `--batch` to 32 and watch VRAM; drop back if you see an allocator
error. Timing observed on CPU was about 150 s per epoch for 1,500 molecules; the
GPU should be an order faster.

## The leak, and the only legitimate use of it

Every one of the 1,213 test spectra occurs as an exact duplicate inside
`train.parquet`, carrying `normalized_smiles` and `inchikey14`; all 400 test
molecules resolve to a unique structure, every match from `enveda-180`.
`python -m dbf2 oracle` recovers that mapping.

This is a defect in the published data. **Do not use it to build a submission.**
Report it to the organisers. Its legitimate use is the reason it is implemented
here: with `DataConfig.exclude_oracle_from_training = True` (the default) those 400
structures and all their spectra are removed from every training split, and the
recovered key then serves as an honest, instrument-matched held-out benchmark that
no split of the training data can provide.

## What is verified, and what is not

`python -m unittest discover -s dbf2/tests` runs 36 tests. The ones that matter:

* **Mass defect survives.** C4H3O+ (67.0178) and C5H7+ (67.0542) fall in one 0.5 Da
  bin; their embeddings differ by 1.82. A 1 mDa separation at m/z 342 survives the
  float32 cast — the phase at a 2 mDa wavelength exceeds 1e6, so it is reduced
  modulo 2*pi in float64 first. Computing it in float32 would erase exactly what
  Module I exists to preserve.
* **Pooling behaves as specified.** Posterior variance falls monotonically as views
  are added (0.495 to 0.140 over one to six views); uncertainty attenuates the
  probability toward 1/2; masked views contribute nothing.
* **The view count is unbounded.** Sufficient statistics are additive across
  chunks to 5e-7, so a molecule with 1,468 spectra needs no truncation.
* **The sampler recovers signal.** On synthetic data with a nonlinear latent
  residual it reaches correlation 0.91 with the truth without view evidence and
  0.99 with it, and improves negative log likelihood from 0.686 to 0.577 against an
  oracle of 0.567. PG draws match the known mean tanh(c/2)/(2c).
* **Adduct algebra is exact** to 1e-5 across all sixteen adducts in the data.

Three regression tests encode bugs found during development and fixed: pruned BART
children remaining in the node list; `tau` diverging when no view evidence is
supplied, which detached theta from its mean; and a mass-scaled nitrogen ceiling
that excluded C11H16N6O, a real test structure, from formula enumeration.

Measured on 120 test molecules against the recovered key:

| | before fixes | after |
|---|---|---|
| true formula present in the enumeration | 73.3% | **99.2%** |
| true formula ranked top-10 | 28.3% | **60.0%** |
| subformula coverage, true vs other formulas | undiscriminating | **0.72 vs 0.30** |
| candidate recall (formula pool unioned with mass) | 0.90 | **1.00** |

**End-to-end execution is verified** on real data: `prepare` (2 row groups, 43,536
molecules, 262,041 views, 15.3 M peaks, balanced folds), `train` for M1 and M2, and
`bayes` all run to completion and write their reports.

**Scientific claims are not verified.** No rung has been trained to convergence on
the full data, so nothing here supports a claim about MRR@25 or about M2 beating M1.
The baseline to beat is the existing spectrum-to-fingerprint model at held-out
MRR@25 = 0.4975, 0.4585 on the timsTOF slice.

## Known limitations

* `score_formulas` uses a deterministic coverage-and-mass-error score, not the
  learned scorer of equation (34). Top-1 formula accuracy is about 5 percent, well
  below SIRIUS or BUDDY. Candidates are therefore unioned with the mass-only pool so
  that an uncertain formula never reduces recall; the formula term downweights,
  it does not exclude.
* Candidate sources C2 (PubChem, COCONUT, LOTUS), C3 (analogue expansion) and C4
  (generated structures) from equation (40) are not implemented. Only C1, the local
  library, is built. Recall against structures outside the training set is
  consequently zero, which is the binding constraint on this task and is why
  `candidate_recall` is reported alongside every MRR.
* Modules VII (fragment generator) and VIII (forward-spectrum likelihood) are
  specified but not implemented; `GeneralizedPosterior` accepts their terms and
  gives them zero weight until supplied.
* `bayes_stage` applies the fitted forest to held-out molecules by re-evaluating the
  retained trees. This is a plug-in approximation, not a full posterior predictive
  over tree structures.
* `theta ~ N(mu, tau^2)` conditions its prior on `H_i`, which is pooled from the same
  views supplying the evidence, so the spectra enter twice — see specification
  section 5.2. The generalized-Bayes reading is used: the fitted `lambda_B` absorbs
  the inflated precision, and calibration is reported rather than assumed. Run the
  population-prior variant as a diagnostic.
