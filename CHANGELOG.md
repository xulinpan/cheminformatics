# Changelog

Notable changes to the code in this repository. The manuscript's own revision
history is in `manuscript/superseded/` and in the git log.

## msms-controls

### 0.1.0 — unreleased

First release. Extracted from the research code so the two controls can be applied
without installing a training stack.

- `quantise_spectrum`, the rounding control, snapping the precursor to the same
  grid as the peaks.
- `quantise_peaks`, the fragment axis alone.
- `absorbed_fraction`, the share of peaks a grid merges away.
- `conditioning_response` and `check_conditioning_parity`, with
  `ConditioningParityError`.
- Pure numpy, typed, 23 tests.

## dbf2

### 2.0.0 — unreleased

- Packaged: the repository had no build metadata and could not be installed.
  Dependency bounds loosened; the exact versions behind the reported runs are kept
  in `requirements-lock.txt`.
- `BinnedEncoder` and `MeanPooling` now consume the acquisition covariates they
  had been accepting and discarding. This was the confound that halved the
  manuscript's headline result: the binned baseline alone could not see the
  collision energy, adduct, polarity or instrument. All M0 runs on both corpora
  were refitted; every previously reported M0 figure is superseded.
- Added rung **M1D**, an attention-free arm over the same rounded peak tokens,
  with the feed-forward widened so it is not also the smaller model.
- Added the rounding control **M1R** (`quantise_mz`, `quantise_merge`) and the
  open-corpus dataset (`--dataset open`).
- Tests: 67 covering representation integrity and covariate reachability, plus a
  guard that the internal quantisation agrees with `msms-controls`.
