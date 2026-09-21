# A metadata-matched binned baseline halves the apparent peak-level advantage in MS/MS substructure prediction

Code, figures and manuscript source for a controlled decomposition of the gain that
peak-level models show over binned baselines in tandem mass spectrometry.

> **Status.** The manuscript in `manuscript/` is a working draft prepared for
> submission to *Journal of Cheminformatics*. It has not yet been peer reviewed.
> Its claims are current as of the open-corpus replication (three seeds, five arms).
> See [Open items](#open-items) for the controls that are still outstanding.

## The question

Nearly every machine-learning method for MS/MS consumes the spectrum as a
fixed-width binned intensity vector. That representation was inherited from
spectral library search, where a bin comfortably wider than the instrument's mass
error is a virtue: it makes matching robust to calibration drift between
laboratories. Structure annotation is an inference problem rather than a matching
problem, and the same argument does not obviously carry over — but the
representation did, and its cost has not been measured.

A 0.5 Da bin spans roughly 1,700 ppm at *m/z* 300 on an instrument delivering about
5 ppm. What it discards is the mass defect, the quantity that distinguishes
C<sub>5</sub>H<sub>7</sub><sup>+</sup> (67.0542) from
C<sub>4</sub>H<sub>3</sub>O<sup>+</sup> (67.0178) — and therefore the quantity that
carries elemental composition.

## The result

Four arms, each differing from the one above it in a single respect, three training
seeds each. The primary corpus is the openly licensed subset of CASMI 2026 — every
spectrum from GNPS, MassBank and MoNA: 414,992 spectra over 52,428 structures,
66 instrument types. Held-out fold 0 is 10,058 molecules; chance macro AUPRC is
0.0085 over 433 scored targets.

| arm | mass axis | encoder/interface | peak pooling | parameters | macro AUPRC | SD over seeds |
|-----|-----------|---------|---------|-----------:|------------:|--------------:|
| M0 | 0.5 Da bins | dense vector, MLP | mean | 1,172,752 | 0.1418 | 0.0013 |
| M1D | 0.5 Da bins | peak tokens, no attention | mean | 714,528 | 0.1590 | 0.0024 |
| M1R | 0.5 Da bins | peak tokens, attention | mean | 712,336 | 0.1567 | 0.0025 |
| M1 | full precision | peak tokens, attention | mean | 712,336 | 0.1709 | 0.0022 |
| M2 | full precision | peak tokens, attention | hierarchical | 829,584 | 0.1888 | 0.0015 |

**M1R is the control that makes the experiment mean anything.** A binned vector and
a peak set are different kinds of object and no architecture reads both, so
abandoning bins forces an architecture change and every published comparison moves
both at once. M1R gives the binned mass axis the peak-set encoder, so the two
separate.

| contrast | isolates | Δ AUPRC | 95% CI | targets improved |
|----------|----------|--------:|--------|-----------------:|
| M1D − M0 | binned token-encoder package † | +0.0172 | [+0.0102, +0.0242] | 58.0% |
| M1R − M1D | attention-bearing vs attention-free † | **−0.0023** | **[−0.0066, +0.0020]** | 54.3% |
| M1R − M0 | rounded token-encoder package † | +0.0149 | [+0.0080, +0.0221] | 66.7% |
| M1 − M1R | **mass-axis precision** | +0.0142 | [+0.0097, +0.0187] | 64.9% |
| M2 − M1 | **hierarchical aggregation** | +0.0179 | [+0.0136, +0.0223] | 63.7% |
| M1 − M0 | joint token-pipeline and mass-axis | +0.0291 | [+0.0208, +0.0371] | 72.5% |

† composite. The arms either side of these three differ in more than the named
factor — in pooling, in feed-forward width, or in how the acquisition covariates
are injected — so the increment belongs to the package, not to one component. Only
the two contrasts in bold change a single thing.

**Half of what a two-arm comparison credits to the spectral representation was the
baseline's missing metadata.** The binned
baseline in this work, and in the published comparisons it stands in for, was the
only arm that could not see the collision energy, adduct, polarity or instrument:
its encoder accepted the covariate vector and discarded it. Giving M0 the same
conditioning every other arm already had raised it from 0.1130 to 0.1418 and cut the
gain a two-arm comparison would report by 50%, from +0.0579 to +0.0291. What
survives splits into +0.0149 for M1R − M0 and +0.0142 for the mass axis — but only
the second is a one-factor contrast.

**Neither remaining increment dominates the other.** The M1R − M0 share of the
M1 − M0 gain is 51.3% on the open corpus and 46.7% on the single-library one, the
two overlap heavily, and their difference changes sign across seeds on both.
Aggregation (+0.0179) exceeds M1R − M0 in every seed on both corpora, which is the
one ordering the paper claims.

**The attention-bearing configuration buys nothing measurable here.** M1D presents
the same rounded peaks as tokens with no attention anywhere in the network and
matches the attention-bearing model (−0.0023, interval straddling zero, positive in
one seed of three). This is a sensitivity comparison, not an attention ablation:
M1D and M1R also differ in pooling and feed-forward width. An exact ablation holding
both fixed has not been run, and until it is, nothing here prices attention.

### The same decomposition on a second, deliberately contrasting corpus

The corpus file is ordered by source library, so a contiguous slice of it selects a
library rather than sampling. The slice used in earlier versions of this work is
99.0% one in-house library and 99% one instrument (timsTOF). Rerunning the whole
ladder there gives the same qualitative answer with a different balance:

| effect | single library (99% timsTOF) | open (66 instruments) |
|---|---:|---:|
| M1R − M0 (composite) | +0.0234 | +0.0149 |
| mass axis | +0.0268 | +0.0142 |
| aggregation | +0.0322 | +0.0179 |
| M1R − M0 share of M1 − M0 | 46.7% | 51.3% |
| M0's gain from conditioning | +0.0142 | +0.0288 |
| peaks absorbed at 0.5 Da | 34.4% | 23.6% |

Both ladders are fitted against the conditioned baseline, so every row is
comparable. The pattern holds on both: M1R − M0 and the mass axis are
indistinguishable, their difference changes sign across seeds, and aggregation beats
M1R − M0 in every seed. The second corpus is a contrasting sensitivity corpus rather
than an independent replication — it is a contiguous in-house slice of the same
CASMI aggregate, run through the same code.
Every effect is smaller on the open corpus, the mass axis roughly halving as the
instrument mix widens from one to 66. Our hypothesis is that a heterogeneous corpus
contains many spectra whose reported *m/z* carries little real precision; that is
untested, and named in the paper as the most informative experiment left undone.

The covariate correction replicates too, and scales the way it should: conditioning
M0 was worth +0.0288 on the corpus with 66 instrument types and 112 adducts, and
+0.0142 on the one with two and 16.

## Repository layout

```
manuscript/      paperA.tex, results_paperA.tex, and the figures and tables they use
src/dbf2/        the model package: preprocessing, encoders, training, evaluation
scripts/         seed/sweep analysis and the figure and table builders
results/         machine-readable numbers behind every figure and table
requirements.txt pinned environment
```

`results/figure_data.json` contains the values plotted in each figure, so the
figures can be checked against the numbers without rerunning anything.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install rdkit                    # not pinned; version-sensitive on some platforms
```

Build the openly licensed subset, then prepare and train. `--dataset open` reads
`data/train_open.parquet` and writes to `dbf2_prepared_open/` and `dbf2_runs_open/`,
so the two corpora never mix:

```bash
python scripts/make_open_subset.py --root .          # -> data/train_open.parquet
python -m dbf2 prepare --root . --dataset open       # stage 0; resumable via --budget

python -m dbf2 ablate --root . --dataset open --rungs M0 M1R M1 M2 --epochs 15
python -m dbf2 ablate --root . --dataset open --rungs M0 M1R M1 M2 --epochs 15 --seed 7
python -m dbf2 ablate --root . --dataset open --rungs M0 M1R M1 M2 --epochs 15 --seed 13
```

Training writes `<rung>_report.json` and `<rung>_test_predictions.npz` per arm.
Then the analysis, figures and tables:

```bash
python scripts/seed_sweep_analysis.py                # single-library corpus
python scripts/make_open_figures.py                  # fig1, fig2, fig3
python scripts/make_open_tables.py                   # tables 1-3
```

The bin-width sweep varies the mass grid at a fixed encoder and parameter count:

```bash
for w in 0.25 0.1 0.05 0.01; do
  python -m dbf2 train --root . --rung M1 --quantise $w --tag W$w --epochs 15
done
```

`python -m pytest src/dbf2/tests` runs the unit tests — 50 tests, no data required.
Fourteen of them cover the rounding control specifically, including that M1R and M1
have identical parameter counts and that their configurations differ in exactly one
field.

### Data

`scripts/make_open_subset.py` currently selects the GNPS, MassBank and MoNA rows
**out of the aggregated CASMI 2026 file**, which the competition organisers
distribute and which is not redistributed here. Those spectra are openly licensed at
source, but reproducing this work today still requires that file. Fetching the
records from GNPS, MassBank and MoNA directly, so no competition download is needed,
is tracked in [Open items](#open-items). `docs/corpus_composition.md` records each
source library's contribution and what is and is not redistributable.

### One implementation detail that matters

Peak *m/z* is embedded with a multi-scale sinusoidal basis whose shortest wavelength
is 2 mDa. At *m/z* 342 the resulting phase exceeds 10<sup>6</sup>, and evaluating it
in single precision destroys exactly the low-order information the embedding exists
to preserve. The phase is therefore computed in float64 and reduced modulo 2π
*before* the cast to float32 (`src/dbf2/encoder.py`, `FourierMZ`). Without that
reduction the representation advantage disappears.

## Open items

Stated here so that anyone reading the code knows what has and has not been
established.

1. **Reproduction still requires the CASMI download.** The subset builder reads the
   aggregated file rather than fetching from GNPS, MassBank and MoNA. A direct
   fetch, or a DOI deposit of the assembled subset, is needed before the work is
   reproducible without a competition account.
2. **No arm was tuned.** An untuned MLP and an untuned Transformer are not
   equidistant from their optima, and the schedule was chosen for the latter. Now
   that a fair baseline has halved the reported gain, a learning-rate sweep for M0
   alone — reporting its best configuration against M1R's untuned one — is the
   single control most likely to move the remaining +0.0291 again.
3. **The reported intervals do not cover seed variance.** Average precision is
   averaged across seeds and then bootstrapped over targets, so the interval
   describes target sampling only; seed spread is reported separately. A
   variance-components treatment with targets and seeds crossed would give one
   interval covering both.
4. **Targets are treated as exchangeable in the bootstrap** though BRICS fragments
   co-occur systematically, so the intervals are narrower than they should be.
5. **The instrument-resolution hypothesis is untested.** Stratifying the mass-axis
   effect by instrument resolution needs no new training — `instrument_family` is in
   `views/spectra.parquet` and the per-molecule predictions are saved — and would
   turn the explanation into a finding.
6. **The bin-width sweep is single-seed and single-corpus**, so only its endpoints
   should be read as findings and its 0.05 Da threshold may not transfer.
7. **Neither composite contrast has been reduced to one factor.** M1R − M0 changes
   the input interface, the network family, the peak-pooling rule *and* the
   mechanism by which acquisition covariates enter (concatenation against adaptive
   layer normalisation). M1R − M1D changes attention, pooling and feed-forward width
   together. Two runs would fix this: a dense baseline conditioned by FiLM/AdaLN
   rather than concatenation, and an attention ablation holding pooling and width
   fixed on both sides.
8. **No arm was tuned, and the schedule was chosen for the token model.** The
   metadata correction — worth more than any contrast in the study — is direct
   evidence M0 was not near its ceiling. A learning-rate and width search for M0
   alone is the single control most likely to move +0.0149 again.
9. **No external baseline, and the endpoint is a proxy.** Every arm here is ours, and
   whether the gain propagates to top-*k* structure retrieval is not shown.
10. **The generality of the covariate confound is not established.** We show that
   *our* binned baseline was denied metadata the peak-set arms received. Whether
   published binned comparators share the defect is the question that decides
   whether this is a finding about the field or about one codebase, and it is
   answerable from released code.

## Citing

See `CITATION.cff`.

## Licence

Code is MIT (see `LICENSE`). The manuscript and figures in `manuscript/` are
© the authors, released under CC BY 4.0.
