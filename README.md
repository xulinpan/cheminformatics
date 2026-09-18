# Full-precision peak representations for MS/MS substructure prediction

Code, figures and manuscript source for a controlled measurement of what it costs
to bin a tandem mass spectrum before handing it to a model.

> **Status.** The manuscript in `manuscript/` is a working draft prepared for
> submission to *Journal of Cheminformatics*. It has not yet been peer reviewed.
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

Two models were trained on the CASMI 2026 corpus, holding data, structural targets,
scaffold-grouped splits, optimiser, schedule and evaluation fixed. The binned arm
was given 63% *more* parameters, so any deficit cannot be attributed to capacity.

| rung | encoder | pooling | parameters | macro AUPRC | macro AUROC | micro AUPRC | Brier |
|------|---------|---------|-----------:|------------:|------------:|------------:|------:|
| M0 | 0.5 Da bins | mean | 1,164,560 | 0.0476 | 0.7832 | 0.2905 | 0.00704 |
| M1 | peak set | mean | 712,336 | 0.1119 | 0.8734 | 0.3842 | 0.00653 |
| M2 | peak set | hierarchical | 829,584 | 0.1460 | 0.8861 | 0.4088 | 0.00639 |

Chance macro AUPRC is 0.0093 over 477 scored targets on 8,692 held-out molecules.

| comparison | change | Δ AUPRC | 95% CI | targets improved |
|------------|--------|--------:|--------|-----------------:|
| M1 − M0 | representation | +0.0644 | [+0.0570, +0.0722] | 89.7% |
| M2 − M1 | aggregation | +0.0341 | [+0.0277, +0.0407] | 72.3% |
| M2 − M0 | both | +0.0984 | [+0.0885, +0.1090] | 94.3% |

The representation change is 1.9× the architectural change measured on the same
molecules with the same targets. Separately, 37.5% of the 93,611 evaluation peaks
that survive preprocessing fall into a 0.5 Da bin already occupied by another peak
from the same spectrum, and are therefore absorbed. (The raw files give 60.7%, but
preprocessing discards 76.6% of those peaks before either arm is trained, so that
figure charges binning for peaks no model receives.)

## Repository layout

```
manuscript/      paperA.tex, paperA.pdf and the figures and tables they use
src/dbf2/        the model package: preprocessing, encoders, training, evaluation
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

1. **The two arms differ in encoder as well as representation** (an MLP over the
   binned vector, a Transformer over the peak set). The clean isolation is rung
   **M1R**, implemented in `src/dbf2/dataset.py` (`quantise_merge`): each peak's
   *m/z* is rounded to its bin centre, collisions are merged, and the result goes
   through the *identical* Transformer at an identical parameter count. The code
   and its tests are in place; **the run itself is still outstanding**, and its
   result determines whether the paper's central claim holds.
2. **One training run per arm.** The reported intervals bootstrap over targets and
   do not capture initialisation variance. Seed replication is outstanding.
3. **The mass-defect mechanism is argued but not ablated.** Removing the explicit
   defect channel and the sub-Dalton wavelengths, with peak identity otherwise
   intact, would test it directly.
4. **The endpoint is a proxy.** BRICS substructure AUPRC is an intermediate; whether
   the gain propagates to top-*k* structure retrieval is not shown here.
5. **Scale.** Experiments use two of twenty-one corpus row groups, roughly a tenth
   of the available data.

## Citing

See `CITATION.cff`.

## Licence

Code is MIT (see `LICENSE`). The manuscript and figures in `manuscript/` are
© the authors, released under CC BY 4.0.
