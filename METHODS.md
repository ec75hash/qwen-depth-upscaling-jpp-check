# Measurement and reproduction methods

## Sources and decoder selection

`models.lock.json` records full Hugging Face checkpoint commits, SHA-256s of config and
safetensors index files, exact decoder namespace prefixes, expected depth, tensor
families, and the exploratory region/period used for reporting. Model depth comes from
`text_config.num_hidden_layers` (or the top-level field for SOLAR). Tensor selection uses
an anchored full-name match under the declared decoder prefix. Auxiliary `mtp.layers.*`
and vision blocks cannot enter the decoder analysis. Configured attention-type coverage
must match selected layer IDs when `layer_types` is available.

The three checkpoints are Qwen3.6-27B (primary), Qwen3.5-9B (negative control), and
SOLAR-10.7B-v1.0 (disclosed-duplication positive control). The smaller Qwen model is a
control for the tested tensor families, not a claim about every parameter or possible
form of shared initialization in that model.

## Byte-range capture

The scanner downloads public config/index files through `huggingface_hub`, then reads
safetensors headers and selected tensor payloads with HTTP byte ranges. A response must
have status 206, the requested Content-Range, and the exact expected byte length.
Non-range responses are rejected before reading the body. Each tensor is recorded with
its original dtype, shape, source shard, absolute payload byte range, header hash, and
payload SHA-256. BF16 payloads are decoded exactly to float32; similarity is computed
with NumPy float32 linear algebra. Full weight shards and raw tensor payloads are not
redistributed. The local tensor cache is excluded from version control and uploads.
When a distributed capture is present, online/cached tensor bytes must also match its
recorded payload hashes for the same checkpoint revision and tensor name.

## Pair and offset statistics

For each family, flatten the tensor at each selected decoder block and normalize by
its L2 norm. The Gram matrix gives every pair cosine. Shapes must agree; zero/nonfinite
norms and unsupported dtypes fail the scan. Self-pairs are omitted.

Offsets are differences of original decoder block IDs, not ranks in a sparse list of
tensor-bearing blocks. Enumerate 1 through depth−1 and omit offsets with zero valid
pairs. Every supported offset remains in `data/offsets.csv`. Rankings and spectrum
figures require at least **six** pairs, to avoid singleton tails dominating a summary.
That threshold is a reporting rule, not a significance test.

At offset k, `all_valid_mean` averages all existing pairs (ℓ, ℓ+k). `inside_mean` uses
only pairs whose two endpoints lie inside the preselected reporting region. The primary
region is 8–55 and its candidate period is 16. The SOLAR region is 8–39; the negative
control uses the entire decoder and has no proposed copied region. Period-32 statistics
are reported separately. Conv1d and key-projection tensor dimensions differ, so their
cosines are complementary measurements rather than directly interchangeable scales.

## Controls and interpretation

The negative-control model uses the same tensor families. The positive control has a
published duplication recipe. Within the primary model, boundary profiles and other
offsets distinguish the localized pattern from generic layer-type repetition. A future
function study should additionally match layer types, perturbation dose, and task inputs.

This is a deterministic checkpoint measurement, not independent samples of model
training. There are no token-level p-values or confidence intervals in this release.
The region was selected during earlier exploration. The regression and release checks
verify numerical plumbing and stated evidence; they do not prove the unpublished
construction history or a causal role for repeated ancestry.

## Offline versus online reproduction

`verify_release.py` checks checksums, namespaces, coverage, and qualitative evidence
conditions against the distributed capture. `make_report.py` regenerates figures and
tables offline. `scan_weights.py` independently fetches the pinned selected tensors and
recomputes the pair statistics online. Tiny floating-point differences across BLAS
implementations may occur. The online capture should be compared numerically with the
release; artifact SHA-256s refer to exact bytes, not a tolerance-based equivalence claim.

## October 12 readout reanalysis

The readout supplement is a separate saved-artifact experiment, not a fresh inference
run. `scripts/import_readout_capture.py` exports only held-out rows from the original
rank table and preserves aggregate targets and sanitized provenance. It excludes the
unrelated exploratory conversation section of the original result file.

`scripts/reanalyze_readouts.py` reconstructs full-vocabulary and filtered rank matrices
for three readouts at layers 0–62, verifies complete/unique coverage, and reproduces
the original region scores, per-layer curves, and offset correlations. Recall@10 first
averages intermediate hits within an item, then items within a task, then the five tasks.
Region scores take the best rank within each region. Offset Spearman correlations use
the same held-out item–intermediate units and common layer anchors for the central
and flanking same-type offsets. Average tied ranks are used; applying log to ranks
does not change their Spearman correlation.

The original item-bootstrap intervals remain in `capture_summary.json`. The main
note/figure use point estimates, which differ slightly from bootstrap resampled means.
The published-convention gap uses filtered J++ and full-vocabulary J-lens; common-filter
gaps are also shown. Split uncertainty, artifact fitting-budget differences, and MLX
4-bit activations are recorded in the supplement. No statistical equivalence test is
claimed. Reanalysis reproduces the saved rank statistics, not the original capture or
the authors' matched-budget BF16 performance comparison.
