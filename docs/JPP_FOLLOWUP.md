# J++ follow-up: completed evidence and open comparison

The architecture scan in this release is complete and independently reproducible.
The following readout work is a separate preliminary analysis and is not included as
a headline result of the architecture dataset.

Earlier local work evaluated the released Qwen3.6-27B J++ lens and the Neuronpedia
J-lens using the authors' items and filtering conventions, with MLX 4-bit forward
activations. It compared lens readouts at every layer and within the three proposed
repeated regions. The comparison used different fitting budgets (64 versus 1000
prompts), an assumed seed-0 fitting/evaluation split that the artifact does not confirm,
and a different forward precision from the authors' BF16 evaluation. It should therefore
be described as a released-artifact comparison, not a matched method replication.

No conclusion that duplication explains or does not explain J++'s gains follows from
those preliminary comparisons. Similar observed blockwise gaps are not an equivalence
test. Smooth cumulative lens matrices need not reflect a local weight-ancestry pattern.

## Specific next request to the authors

Request the corresponding Qwen3.6-27B J-lens fitted on the paper's 64-sequence budget,
and the fitting/evaluation item split used for the released J++ expert weights, if not
already publicly available. Reuse the released evaluation code at a pinned commit.

## Proposed matched analysis

1. Confirm block-output indexing, checkpoint/lens versions, item correctness filtering,
   scoring positions, vocabulary filtering, and the expert-weight fitting split.
2. Capture BF16 activations using the authors' wrapper. Compare matched-budget lenses
   at the same positions, with the same items and scoring definitions.
3. Report absolute depth and relative slot within each inferred repeated region.
4. Compare corresponding intervention windows individually and jointly. Include
   non-aligned same-type windows and controls for perturbation magnitude and number
   of intervened blocks; retain the original naming/avoiding or focus/suppress tasks.

The email can communicate the architecture finding before these experiments run.
There is no requirement to complete a causal mechanism study to share a reproducible
structural observation with researchers using the checkpoint.
