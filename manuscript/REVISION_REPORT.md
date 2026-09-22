# Editorial revision report

Manuscript: *Metadata matching halves the apparent advantage of a full-precision
peak-token model in MS/MS substructure prediction* (formerly *A metadata-matched
binned baseline halves the apparent peak-level advantage in MS/MS substructure
prediction*).

Superseded draft retained at `manuscript/superseded/paperA_pre_editorial_revision.tex`.

---

## 1. Editorial summary

The manuscript's scientific message is unchanged: after acquisition metadata are
matched across comparators, the apparent advantage of a full-precision peak-token
representation falls by about half, and the remainder must be decomposed by
within-architecture controls rather than attributed to representation wholesale.
No reported number has been altered. Three changes do most of the work.

**The comparator is now treated as the paper's own weakest point, in the open.**
The manuscript previously noted that no arm was tuned. It now states that this is
not a symmetric choice — the schedule was chosen for the token model and inherited
by the dense baseline — and prespecifies an M0-only tuning experiment with the
split fixed, selection on fold 1, and a single evaluation on fold 0. The
manuscript says in advance what happens to the decomposition if the tuned
baseline moves it.

**Uncertainty is now reported on three axes instead of one.** The
target-resampling bootstrap is retained and correctly scoped: it describes
heterogeneity across targets at a fixed held-out set, and nothing else. A
scaffold-cluster bootstrap over held-out structures has been added, resampling
scaffold groups rather than molecules so that molecule-level dependence is
preserved. A Methods subsection separates seed variability, target-resampling
uncertainty and test-set uncertainty, and neither interval is called a confidence
interval for the population of MS/MS tasks.

**M2 is now specified mathematically.** It was described verbally while
contributing the single largest one-factor increment in the study. The
implementation was read and transcribed rather than idealised: the manuscript now
gives the acquisition-dependent reliability, how positivity is enforced, the
normalised representation pooling, the Gaussian-conjugate evidence pooling, the
logistic–Gaussian output map, the handling of missing metadata, and the fact that
all parameters are trained jointly and that the sufficient statistics are additive
over spectra.

Two analyses requested as markers turned out to be runnable from saved artefacts
and were executed rather than deferred: the cross-fold molecular identity audit
(clean on both corpora) and the scaffold-cluster bootstrap. Three items remain
genuinely outstanding and are marked in red in the manuscript.

Terminology has been made consistent throughout: *rounded token-encoder package*
rather than encoder effect, *configuration sensitivity comparison* rather than
attention effect, *sensitivity corpus* rather than replication. Every contrast in
the Results carries an explicit ONE-FACTOR or COMPOSITE label, and causal language
appears only on the former.

---

## 2. Point-by-point revision table

| Issue | Current problem | Revision performed | New experiment required? | Location |
|---|---|---|---|---|
| **A. Dense baseline strength** | M0 inherited the token model's schedule; no tuning. Fatal exposure for a paper about comparator fairness | Asymmetry stated explicitly; prespecified M0-only grid (LR ×3, width ×3, dropout ×2), split fixed, selection on fold 1, single fold-0 evaluation; instruction to recompute all M0-involving contrasts if the decomposition moves | **Yes** — `[REQUIRES NEW EXPERIMENT: tuned M0]` | Methods "Hyperparameter selection, and a known asymmetry"; Limitations; Discussion |
| **B. Uncertainty quantification** | Single bootstrap over targets, implicitly treated as the interval | Retained and rescoped; scaffold-cluster bootstrap run (2,000 draws, 3,784 clusters), estimator verified against all seven point estimates, results in Table 5 with the bootstrap mean shown beside each point estimate; Methods subsection separating the three sources | No — complete | Methods "Evaluation and three sources of uncertainty"; Results §"Test-set uncertainty", Table 5 |
| **C. M2 formalisation** | Largest one-factor effect described only verbally | Full equations transcribed from the implementation, including two-level aggregation, softplus+floor positivity, normalised weights, additive sufficient statistics, joint training | No | Methods "Multi-spectrum aggregation in M2", Eqs. (2)–(8) |
| **D. Attention ablation** | M1R−M1D presented with attention prominence despite three simultaneous differences | Renamed throughout; removed from Conclusion emphasis; exact ablation specified (identical pooling, width, protocol; operator toggled) | **Yes** — `[REQUIRES NEW EXPERIMENT: exact attention ablation]` | Methods "An exact attention ablation"; Results §4; Abstract |
| **E. Provenance / reproducibility** | "should ultimately be achieved" — prospective | Rewritten into five concrete subsections: code, environment, primary corpus, filtering and identity, artefacts, sensitivity corpus; per-record licence filtering noted for all three source libraries | **Yes** — `[REQUIRES REPOSITORY ARCHIVE / DOI]` | Availability of data and materials |
| **F. Identity and split validation** | Identity rule unstated; split asserted not verified | Identity defined as InChIKey first block with explicit treatment of stereoisomers, tautomers, salts, protonation and cross-library duplicates; audit **run**: 0/50,287 identifiers and 0/19,648 scaffold groups span folds; scaffold grouping distinguished from chemical-space independence | No — audit executed | Methods "Molecular identity and the fold assignment"; "Split-integrity audit" |
| **G. Literature positioning** | Thin; risked reading as a leaderboard entry | Introduction expanded with metadata-conditioning and multi-spectrum aggregation strands; explicit statement that external systems are not benchmarked and why; dedicated Discussion subsection | No | Introduction ¶4–7; Discussion "Relation to other work" |
| **H. Title** | Named the baseline, not the experiment | Changed to the proposed wording; scoped to "a full-precision peak-token model", not all peak-level models | No | Title |
| **I. Abstract** | "reproduces the pattern"; ended on model comparison | "shows a qualitatively similar pattern"; ends on experimental design; all eight required numbers retained; 342 words total with the contribution statement (limit 350), 3 sentences | No | Abstract |
| **J. Results structure** | Organised by arm name | Reorganised into eight question-led subsections; every contrast labelled ONE-FACTOR or COMPOSITE; causal language confined to one-factor rows | No | Results §1–8 |
| **K. Figures** | Fig. 2 crowded, colour-only encoding | Marker shape added (circle = one-factor, square = composite), interval caps, labels repositioned above means; Fig. 1 seeds retained; Fig. 3 caption states it is descriptive; sweep table gains a Status column marking single-seed runs | No | Figs. 1–3; Table 4 |
| **L. Limitations** | Flat list | Six categories: experimental design, inference, ablation, data, scope, reproducibility | No | Discussion "Limitations" |
| **M. Claim strength** | "encoder effect", "attention effect", "replication" | Terminology hierarchy applied throughout, including tables and figure captions | No | Throughout |
| **N. Conclusion** | Single narrative | Three explicit levels: confounding, retained mass-precision effect, comparator and aggregation of similar magnitude; no claim that mass precision is unimportant or that effects are causally equal | No | Conclusion |

---

## 3. Analyses still required before submission

Three. Everything else in the manuscript is complete and its numbers are
reported.

**1. Tuned M0 baseline** — `[REQUIRES NEW EXPERIMENT: tuned M0]`
Prespecified in Methods. Grid: learning rate ∈ {3×10⁻⁴, 10⁻³, 3×10⁻³}, hidden
width ∈ {256, 512, 1024}, dropout ∈ {0.1, 0.3}, 18 configurations, scaffold split
unchanged, folds 2–4 for fitting, fold 1 for selection, fold 0 evaluated once for
the selected configuration under three seeds. Tests whether the residual M1R−M0
difference survives a stronger dense comparator. If the decomposition moves, every
M0-involving contrast must be recomputed; M1−M1R and M2−M1 are unaffected by
construction. Roughly 21 training runs.

**2. Exact attention ablation** — `[REQUIRES NEW EXPERIMENT: exact attention ablation]`
Identical input, peak pooling, feed-forward width, parameter count as closely as
the operator allows, and training protocol on both sides, with only self-attention
toggled. Note the attention-bearing arm must use masked-mean peak pooling for this,
not a learned token. Until it is run, no statement in the paper prices attention.
Six training runs.

**3. Repository archive and DOI** — `[REQUIRES REPOSITORY ARCHIVE / DOI]`
A tagged GitHub release archived to Zenodo, with the DOI inserted in Availability
of data and materials and in `CITATION.cff`. Procedure documented in
`docs/releasing.md`.



### Completed during this revision (no longer outstanding)

- **Cross-fold molecular identity audit.** Executed on both corpora.
  Primary: 0 of 50,287 identifiers and 0 of 19,648 scaffold groups span folds;
  all 372,200 spectra inherit their structure's fold. Sensitivity corpus:
  0 of 43,536 and 0 of 13,620. `scripts/split_integrity_audit.py`.
- **Scaffold-cluster bootstrap.** Complete: 2,000 draws over 3,784 scaffold
  clusters, reported in Table 5. The estimator was verified first — it reproduces
  all seven point estimates exactly.

  A first attempt produced intervals that were quietly wrong, and the reason is
  worth recording because it is easy to repeat. The reported contrasts average each
  target's average precision *across seeds* and then difference. Averaging the three
  seeds' *probabilities* first and computing one AP is a different quantity — a
  three-seed ensemble, a better model than any single seed, which improves the arms
  unevenly. That alone moved M1R−M0 from +0.0149 to +0.0327 before any resampling.
  A second, smaller error admitted targets with fewer than five positives in a
  resample. Both are fixed, and a centring check now runs automatically.

  The finished run shows the two single-factor interventions centring to within
  0.0009 and 0.0004 of their point estimates, and every composite contrast biased
  upward by +0.0026 to +0.0073 — more the more components it bundles. This is
  reported in the manuscript rather than smoothed over: resampling with replacement
  duplicates structures, which sharpens macro AUPRC, and that sharpening cancels
  between arms that rank similarly but not between a dense perceptron and a
  Transformer.

### Derivable from saved predictions, not yet run

- **Instrument-resolution stratification** of the M1−M1R effect. Would convert the
  paper's stated hypothesis about corpus heterogeneity into a tested result. Needs
  no retraining: `instrument_family` is in `views/spectra.parquet` and per-molecule
  predictions are saved.
- **Conditioning-mechanism control:** a dense baseline conditioned through adaptive
  normalisation rather than concatenation, to test whether the +0.0149 M1R−M0
  increment depends on injection mechanism. This one does require training.

---

## 4. Pre-submission checklist

**Numbers and internal consistency**
- [x] Every reported value cross-checked against the analysis artefacts; none altered
- [x] Arm means, seed SDs, parameter counts consistent between Table 1, figures and text
- [x] All seven contrasts in Table 2 match `seed_analysis_open.json`
- [x] Two-corpus table matches `seed_sweep_analysis.json`
- [x] Per-seed ordering figures quoted in text match the ordering checks
- [x] Abstract retains all eight required quantitative findings

**Compliance**
- [x] Abstract + Scientific contribution 342 words (limit 350)
- [x] Scientific contribution 3 sentences (limit 3)
- [x] Keywords: 8 (range 3–10)
- [x] Declarations present: acknowledgements, contributions, funding, availability, competing interests
- [ ] DOI inserted in Availability and `CITATION.cff` — blocked on item 3 above

**Claim discipline**
- [x] "encoder effect" absent
- [x] "attention effect" absent
- [x] "external validation" / "independent replication" absent
- [x] Every Results contrast labelled ONE-FACTOR or COMPOSITE
- [x] Causal language restricted to one-factor interventions
- [x] No claim that mass precision is unimportant
- [x] No claim of equal causal effect between architecture and mass precision
- [x] No universal generalisation beyond endpoint and corpora

**Method completeness**
- [x] Molecular identity rule stated, with stereoisomer/tautomer/salt/duplicate handling
- [x] Split integrity verified and reported, not asserted
- [x] M2 written out mathematically, matching the implementation
- [x] Missing-metadata handling stated
- [x] Three uncertainty sources separated
- [x] Implementation stack, versions and hardware stated
- [x] Provenance of every arm stated (single codebase, no external comparator)

**Outstanding before submission**
- [ ] Tuned M0 experiment run and all M0-involving contrasts updated
- [ ] Exact attention ablation run, or its marker and the associated text removed
- [ ] Zenodo DOI minted and inserted
- [ ] All three red `[REQUIRES …]` markers resolved and the `xcolor` macros removed
