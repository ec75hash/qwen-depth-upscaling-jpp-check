# Corrections and scope of the architecture release

The original exploratory `dense_period_scan.py` was generalized into this package's
`scripts/scan_weights.py`. The release addresses these concrete issues:

1. **Depth versus tensor coverage.** A “48 layers” conv1d headline counted DeltaNet
   tensors in a 64-block model. The release reports 64 total decoder blocks and 48
   tensor-bearing blocks separately.
2. **Offset cutoff.** Searching `range(1, len(tensor_layers))` missed valid offsets
   in sparsely represented families. The release searches the full decoder depth
   using original block IDs.
3. **Empty means.** Offsets without valid pairs produced NaNs that made ranking
   unreliable. Pairless offsets are now absent, and every reported offset has a count.
4. **Auxiliary-head contamination.** A broad `layers.(number)` name match included
   `mtp.layers.0.self_attn.k_proj.weight` in the decoder family. Anchored namespace
   selection now gives 16 decoder attention key projections, excluding the auxiliary head.
5. **Small-count ranking.** Offsets with one or two pairs may produce unstable top-offset
   summaries. The release ranks offsets with at least six pairs while retaining all
   supported offsets in the data.
6. **Mean definition.** Offset means across all valid pairs are distinguished from
   means inside the inferred copied region. The earlier phrase “0.57 averaged over the
   block” blurred those quantities.
7. **Second peak.** “Essentially zero at any other offset” omitted the period-32 peak.
   Both period-16 and period-32 results are reported.
8. **Readout grid.** The J++ grid samples two recurring slots, 8/24/40 and 16/32/48,
   with layer 56 outside the region. It does not sample one identical slot throughout.
9. **Lens comparison.** The existing 64-prompt J++ versus 1000-prompt Neuronpedia J-lens
   comparison is an artifact comparison. It cannot isolate a method effect or establish
   a matched reproduction of the J++ paper. A 4-bit forward pass and unconfirmed split
   are additional differences.

The primary conv1d offset-16/32 pattern and boundary profile survive the corrected
enumeration. Excluding the auxiliary key projection changes the attention-family
background/offset coverage; see the newly captured [results](RESULTS.md) for the release's
exact decoder-only values. Older working logs remain historical artifacts and are not
silently presented as corrected captures.

These corrections concern the architecture scan and J++ follow-up framing. The separate
MoE availability study has its own intervention correction and is not republished here.

## Packaging history

The initial Hugging Face snapshot is tagged v0.1.0. The matching GitHub/Hugging Face
release is v0.1.1: the optional CI workflow was moved into `docs/` because the publishing
GitHub OAuth token does not permit installing workflows. Scientific data, scanner code,
and figures are unchanged. The original Hub tag is preserved.
