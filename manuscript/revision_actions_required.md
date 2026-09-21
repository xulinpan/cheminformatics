# Revision actions for the strongest Journal of Cheminformatics resubmission

The revised manuscript removes or narrows claims that were not isolated by the current experiments. No new numerical result has been invented.

## Already fixed in the revised manuscript

- Changed the title and framing from a universal peak-level/representation claim to MS/MS BRICS substructure prediction.
- Distinguished the 414,992-spectrum source corpus from the 372,200 spectra retained after preprocessing.
- Replaced "fair baseline" with the more precise "metadata-matched baseline" where conditioning mechanisms differ.
- Re-labelled M1R-M0 as a rounded token-encoder *package* effect rather than a pure encoder effect.
- Re-labelled M1R-M1D as a sensitivity comparison rather than a pure attention effect because pooling and feed-forward width differ.
- Softened "attention contributes nothing" to "no measurable benefit from the attention-bearing configuration in this comparison."
- Softened seed-based ranking claims and explicitly described the bootstrap intervals as target-resampling intervals that do not include split uncertainty or target dependence.
- Re-described the single-library corpus as a contrasting sensitivity corpus, not independent replication.
- Converted the instrument-resolution explanation back to an explicit hypothesis.
- Removed the inconsistent peak-absorption column from the bin-width table rather than guessing whether 34.4% or 37.5% is correct.
- Narrowed extrapolation to CSI:FingerID, MIST, DreaMS and TransExION.

## Two reruns that would materially strengthen the paper

1. **Exact attention ablation.** Run an attention-bearing and attention-free rounded-token pair with identical pooling, feed-forward width, parameterisation as far as feasible, training schedule, seed set and acquisition conditioning. This is needed before using language such as "the effect of attention."

2. **Conditioning-mechanism control for M0.** Add a dense baseline whose acquisition covariates modulate hidden layers (for example FiLM/AdaLN-style conditioning) rather than entering only by concatenation. This tests whether M1R-M0 is stable to the way metadata are injected.

A high-value analysis that may not require retraining is to stratify the saved M1-M1R held-out differences by instrument-resolution class. That would turn the current instrument-resolution hypothesis into a directly tested result if the necessary metadata and predictions are already saved.
