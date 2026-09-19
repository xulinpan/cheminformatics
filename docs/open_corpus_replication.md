# Replication on the open corpus

The decomposition was rerun end to end on the GNPS + MassBank + MoNA subset
(`--dataset open`): 50,287 molecules, 372,200 views, 24.6M peaks, 66 instrument
types, three seeds per arm. Held-out fold 0 is 10,058 molecules; chance macro AUPRC
is 0.0085 over 433 scored targets.

## Arms

| arm | mass axis | encoder | macro AUPRC | SD over seeds |
|-----|-----------|---------|------------:|--------------:|
| M0 | 0.5 Da bins | MLP | 0.1130 | 0.0012 |
| M1R | 0.5 Da bins | Transformer | 0.1567 | 0.0025 |
| M1 | full precision | Transformer | 0.1709 | 0.0022 |
| M2 | full precision | Transformer + hierarchical | 0.1888 | 0.0015 |

## Effects

| contrast | isolates | Δ AUPRC | 95% CI | targets improved | SD over seeds |
|---|---|---:|---|---:|---:|
| M1R − M0 | encoder | +0.0438 | [+0.0364, +0.0517] | 79.2% | 0.0030 |
| M1 − M1R | mass axis | +0.0142 | [+0.0097, +0.0188] | 64.9% | 0.0004 |
| M2 − M1 | aggregation | +0.0179 | [+0.0137, +0.0222] | 63.7% | 0.0009 |
| M1 − M0 | both | +0.0579 | [+0.0494, +0.0667] | 81.8% | — |
| M2 − M0 | all three | +0.0758 | [+0.0656, +0.0862] | 83.4% | — |

## What changed against the single-library corpus

| | single library (timsTOF) | open (66 instruments) |
|---|---:|---:|
| encoder | +0.0377 | +0.0438 |
| mass axis | +0.0268 | +0.0142 |
| aggregation | +0.0322 | +0.0179 |
| encoder share of M1 − M0 | 58.5% | 75.5% |
| mass axis vs aggregation | reverses under 1 of 3 seeds | consistent, 3 of 3 |

Three things follow.

**The finding replicates and strengthens.** The encoder's share of the gain a
two-arm comparison would credit to the representation rises from 58% to 76%.

**The ordering the first corpus could not resolve is resolved here.** On the single
library, mass axis versus aggregation reversed under one seed and we declined to
rank them. On the open corpus all three pairwise orderings hold in all three seeds:
encoder > aggregation > mass axis. The mass-axis effect is also the most stable of
the three across seeds (SD 0.0004), so the ranking is not an artefact of noise.

**Mass precision is worth about half what it was.** +0.0268 falls to +0.0142, while
still excluding zero. One hypothesis fits the change: the single-library corpus is
99% timsTOF, a high-resolution instrument on which every reported m/z carries real
precision. The open corpus spans 66 instrument types of varying resolution, and on
the low-resolution ones there is less precision to preserve, so the average benefit
of preserving it falls. That is testable by stratifying the mass-axis effect by
instrument resolution, which we have not done.

Absolute performance is higher everywhere despite a lower chance level (0.0085 vs
0.0093), consistent with 1.58x the spectra and 7.4 views per molecule against 6.0.
