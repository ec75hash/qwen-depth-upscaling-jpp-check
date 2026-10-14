# October 12 J++ readout evidence

This supplement publishes a saved-rank reanalysis of the author's October 12 experiment:
329 locally held-out items, 526 (item, intermediate) units, 63 lens layers, and three
readouts (J++, Neuronpedia J-lens, and logit lens). It contains 99,414 rank rows.

- `held_out_ranks.csv`: the held-out subset of the original long rank table. Fields are
  lens, layer, eval, item identifier, intermediate label, split, rank_full, rank_filtered.
  These are scores on public evaluation items, not private conversation transcripts.
- `capture_summary.json`: original aggregate statistics, extracted from the saved result
  file without unrelated exploratory conversations. Includes the original item-bootstrap
  intervals; their resampled means can differ slightly from the point estimates.
- `provenance.json`: source-file hashes, lens hashes, authors' code commit, cohort/split
  rules, quantized-model config hash, head-fidelity checks, and reproduction scope.
- `reanalysis.json`: regenerated point estimates, per-layer curves, boundary deltas,
  and same-type offset controls. Run `python scripts/reanalyze_readouts.py` to recreate.

Scoring takes the best rank over a layer set for each intermediate; averages intermediate
hits within each item, then items within each task, then macro-averages five tasks.
The main convention compares filtered J++ with full-vocabulary J-lens; common-filter
comparisons are reported alongside it. Offset correlations use the same held-out units
and common source-layer anchors for the central offset and its same-type neighbors.

“Held-out” refers to the locally reconstructed seed-0 split. Whether the released J++
expert weights used that exact split is unconfirmed. The J++ and Neuronpedia J-lens
artifacts have different fitting budgets (64 versus 1000 prompts). Forward activations
were captured with MLX 4-bit, not the authors' BF16 stack. The original head-fidelity
pilot checked the saved unembedding calculation against the quantized model head, not
against BF16 model activations. These limitations qualify the comparison, while the
reported block gains, boundary deltas, and offset correlations remain measured results.

Reanalysis requires only the released ranks and NumPy. Re-executing the activation
capture or fitting lenses is outside this release's reproduction command. No activations,
quantized weights, lens weights, or unrelated exploratory prompt captures are included.
