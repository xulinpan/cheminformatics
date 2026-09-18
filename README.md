# Peak-level MS/MS: most of the advantage is the encoder, not the representation

Code, figures and manuscript source for a controlled decomposition of the gain that
peak-level models show over binned baselines in tandem mass spectrometry.

> **Status.** The manuscript in `manuscript/` is a working draft prepared for
> submission to *Journal of Cheminformatics*. It has not yet been peer reviewed.
> Its claims are current as of the three-seed replication and bin-width sweep.
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

Four arms on the CASMI 2026 corpus, each differing from the one above it in a
single respect, three training seeds each, on 8,692 held-out molecules. Chance
macro AUPRC is 0.0093 over 477 scored targets.

| arm | mass axis | encoder | pooling | parameters | macro AUPRC | SD over seeds |
|-----|-----------|---------|---------|-----------:|------------:|--------------:|
| M0 | 0.5 Da bins | MLP | mean | 1,164,560 | 0.0478 | 0.0006 |
| M1R | 0.5 Da bins | Transformer | mean | 712,336 | 0.0854 | 0.0014 |
| M1 | full precision | Transformer | mean | 712,336 | 0.1122 | 0.0039 |
| M2 | full precision | Transformer | hierarchical | 829,584 | 0.1444 | 0.0016 |

M1R is the control that makes the experiment mean anything: it gives the binned
mass axis the peak-set encoder, so the encoder and the mass axis separate.

| contrast | isolates | Δ AUPRC | 95% CI | targets improved |
|----------|----------|--------:|--------|-----------------:|
| M1R − M0 | encoder | +0.0377 | [+0.0331, +0.0424] | 90.6% |
| M1 − M1R | mass axis | +0.0268 | [+0.0221, +0.0317] | 74.8% |
| M2 − M1 | aggregation | +0.0322 | [+0.0275, +0.0370] | 79.5% |
| M1 − M0 | encoder and mass axis | +0.0645 | [+0.0576, +0.0713] | 92.5% |

**58% of what a two-arm comparison would attribute to the spectral representation
is the encoder.** The encoder effect is the largest of the three and exceeds both
others in all three seeds. The mass axis and aggregation are of comparable size and
their ordering reverses under one seed, so we do not rank them.

A bin-width sweep at fixed encoder and parameter count shows precision paying only
below 0.05 Da: 0.5, 0.25 and 0.1 Da all sit within noise of each other, 0.01 Da
closes 65% of the distance to full precision, and full precision closes the rest.
Absorption of peaks into occupied bins falls from 37.5% at 0.5 Da to 7.5% at
0.01 Da — the two curves do not track each other, so peak absorption alone does not
account for the cost.

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
pip install rdkit                     # not pinned; version-sensitive on some platforms

python -m dbf2 prepare  --root <data-root>
python -m dbf2 train    --root <data-root> --rung M0    # binned control
python -m dbf2 train    --root <data-root> --rung M1    # full-precision peak set
python -m dbf2 evaluate --root <data-root>
python -m dbf2 figures  --root <data-root>
```

`python -m pytest src/dbf2/tests` runs the unit tests (36 tests, no data required).

The CASMI 2026 corpus is distributed by the competition organisers and is not
redistributed here.

### One implementation detail that matters

Peak *m/z* is embedded with a multi-scale sinusoidal basis whose shortest wavelength
is 2 mDa. At *m/z* 342 the resulting phase exceeds 10<sup>6</sup>, and evaluating it
in single precision destroys exactly the low-order information the embedding exists
to preserve. The phase is therefore computed in float64 and reduced modulo 2π
*before* the cast to float32 (`src/dbf2/encoder.py`, `FourierMZ`). Without that
reduction the representation advantage disappears.

## Open items

The manuscript's own limitations, stated here so that anyone reading the code knows
what has and has not been established:

1. **Three seeds is not enough to rank the mass axis against aggregation.** It is
   enough to establish that the encoder effect is the largest and that all three
   components are non-zero. Ranking the smaller two needs more replicates.
2. **The bin-width sweep is single-seed**, so its middle is unresolved; only its
   endpoints are separated by more than the seed spread.
3. **M1R does not control for dense versus sparse encoding.** It gives the binned
   mass axis the better encoder, which isolates precision, but a Transformer over
   bin-indexed tokens is not a perceptron over a dense vector. Some of the encoder
   effect may be that switch. A fifth arm would separate them.
4. **The endpoint is a proxy.** BRICS substructure AUPRC is an intermediate; whether
   the gain propagates to top-*k* structure retrieval is not shown here.
5. **Scale.** Experiments use two of twenty-one corpus row groups, roughly a tenth
   of the available data.
6. **No external baseline.** Every arm here is ours. A published method on the same
   split would anchor the absolute values.

## Citing

See `CITATION.cff`.

## Licence

Code is MIT (see `LICENSE`). The manuscript and figures in `manuscript/` are
© the authors, released under CC BY 4.0.
