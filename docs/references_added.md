# References added, and what each one does

Seven peer-reviewed references were added after a literature check. They are listed
here with the job each does in the manuscript, and with what still needs verifying.

| reference | venue | role in the paper |
|---|---|---|
| Giné R, Pérez-López I, Badia JM, Capellades J, Yanes O (2026) | J Am Soc Mass Spectrom 37:1550–1561 | Closest prior work. Benchmarks featurization strategies for structure annotation across a large model grid. Cited in the Introduction to position what this paper adds: a benchmark measures each featurization bundled with the encoder that reads it, which estimates the joint effect at many points rather than separating it at one. |
| de Jonge NF et al. (2026) | Nat Commun 17:2483 | MS2DeepScore 2.0. Uses 0.1 Da bins and reports that finer bins did not improve performance — which **disagrees with our sweep below 0.05 Da**. Addressed head-on in the Discussion rather than cited in passing. |
| Bushuiev R et al. (2024) | NeurIPS Datasets and Benchmarks | MassSpecGym. Cited in Methods to place our scaffold-grouped split against community practice on generalisation-demanding splits. |
| Kind T, Fiehn O (2007) | BMC Bioinformatics 8:105 | Establishes how the width of the mass window governs the number of surviving candidate formulas. Cited in Results to ground the 328-formulas-per-bin figure in something other than our own enumeration. |
| Beck AG et al. (2024) | ACS Meas Sci Au 4:233–246 | Review of ML for mass spectrometry; surveys the direction of travel away from binning. |
| Wang Y, Wei L, Liu L, Xu H, Ling H (2026) | J Cheminform 18:114 | LLMs on text-serialised peak lists. Evidence that how a spectrum is presented to a model is an open question, not a settled convention. |
| Tan X (2025) | J Cheminform 17:103 | Discretisation onto a fixed grid persists outside MS/MS (IR/UV/NMR into a 66×66 array). Shows the convention is broader than the binning literature. |

The bibliography now runs to 14 references, five of them in *Journal of
Cheminformatics* (MetFrag, MS2DeepScore, TransExION, Wang, Tan).

## Needs verification before submission

The full text of Giné et al. was not accessible during this check — the publisher
and PubMed both returned access challenges. The citation itself is confirmed against
Crossref (authors, journal, volume, issue, pages, year), but the characterisation of
their design in our Introduction rests on the abstract and secondary summaries.
**Read the paper and confirm that description before submitting**, particularly the
claim that each featurization is evaluated with its own encoder. If they do hold an
architecture constant across featurizations, our novelty framing needs revising
rather than merely softening.
