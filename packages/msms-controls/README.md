# msms-controls

Two controls for attributing gains in MS/MS spectral-representation studies.
Pure numpy — no deep-learning framework, no dataset layout, no trained weights.
Apply them to whatever model you already have.

```bash
pip install msms-controls
```

## Why

When a peak-level model beats a binned baseline, the gain is usually discussed as
evidence about the spectral representation. But a binned intensity vector and a
sparse peak set are different kinds of object and no architecture reads both, so
abandoning bins forces a change of encoder. The comparison moves the
representation and the architecture together, and often the acquisition metadata
as well. The increment belongs to the whole package unless the design contains an
intervention that changes one component.

These are the two interventions a controlled decomposition of one such comparison
found necessary. In that study they moved the headline: half of what a two-arm
comparison credited to the spectral representation turned out to be metadata the
baseline had never been given, and the remainder split about evenly rather than
favouring the representation.

## The rounding control

You cannot feed a binned vector to a peak encoder. You can go the other way:
degrade the mass axis in place and pass the result through the encoder you already
have, so the representation moves and the architecture does not.

```python
from msms_controls import quantise_spectrum

mz, intensity, precursor = quantise_spectrum(mz, intensity, precursor, width=0.5)
```

Peaks are snapped to their cell centres on a `width` grid and colliding peaks are
merged with the accumulator your binned baseline uses (`merge="sqrt"` by default,
matching a square-root-intensity binned vector).

**Use `quantise_spectrum`, not `quantise_peaks`.** It also snaps the precursor.
Most peak encoders derive a neutral loss as `precursor - mz`; leave the precursor
at full precision and that subtraction hands every token the precursor's mass
defect, so the arm meant to carry only binned mass information silently regains a
full-precision channel. The control then understates the value of the mass axis by
an amount nothing in the training curves will reveal.

## The conditioning-parity check

Arms of an ablation are often given side information — collision energy, adduct,
polarity, instrument. An arm that accepts those inputs and ignores them does not
crash, does not warn, and produces plausible numbers. Any contrast measured
against it also prices the information it could not see.

```python
from msms_controls import check_conditioning_parity

check_conditioning_parity(
    {"binned": predict_binned, "peaks": predict_peaks},
    {"collision_energy": 10.0, "adduct": "[M+H]+"},
    {"collision_energy": 90.0, "adduct": "[M+Na]+"},
)   # raises ConditioningParityError, naming every arm whose output does not move
```

Each value is a callable mapping conditioning variables to an output array, with
the spectrum held fixed between the two calls.

**If your conditioning path is zero-initialised — FiLM and adaptive layer-norm
usually are — perturb it off initialisation first**, or an untrained model passes
vacuously whether or not it is wired correctly.

## Measuring what a grid costs

```python
from msms_controls import absorbed_fraction

absorbed_fraction(mz, width=0.5)   # share of peaks landing in an occupied cell
```

Computable before any model is fitted. Report it on the same spectra the effects
are measured on; the figure moves with the subset.

## API

| function | what it does |
|---|---|
| `quantise_spectrum(mz, intensity, precursor_mz, width, *, merge="sqrt")` | the rounding control, precursor included |
| `quantise_peaks(mz, intensity, width, *, merge="sqrt")` | the fragment axis alone |
| `absorbed_fraction(mz, width)` | share of peaks merged away by the grid |
| `conditioning_response(predict, setting_a, setting_b)` | how far one arm moves on metadata alone |
| `check_conditioning_parity(arms, setting_a, setting_b, *, atol=1e-6)` | raises `ConditioningParityError` naming unresponsive arms |

## Citing

Pan X, Wang C. *A metadata-matched binned baseline halves the apparent peak-level
advantage in MS/MS substructure prediction.* See `CITATION.cff` in the
[repository](https://github.com/xulinpan/cheminformatics).

## Licence

MIT.
