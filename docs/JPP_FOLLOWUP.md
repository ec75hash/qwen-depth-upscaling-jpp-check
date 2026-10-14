# J++ follow-up: measured readout null and open intervention question

The weight-ancestry result and the October 12 readout check are both included in the
main note. The latter is a measured negative result on the authors' released instrument:
the weight pattern is not mirrored by a distinct periodic J++ readout signature in
this capture. The [saved-rank supplement](../data/jpp_readouts/README.md) makes the
statistics reproducible without running a model.

Earlier local work evaluated the released Qwen3.6-27B J++ lens and the Neuronpedia
J-lens using the authors' items and filtering conventions, with MLX 4-bit forward
activations. It compared lens readouts at every layer and within the three proposed
repeated regions. The comparison used different fitting budgets (64 versus 1000
prompts), an assumed seed-0 fitting/evaluation split that the artifact does not confirm,
and a different forward precision from the authors' BF16 evaluation. It should therefore
be described as a released-artifact comparison, not a matched method replication.

The observed gain is about +33/+35/+33 pp across the three regions under the paper's
reporting convention. J++ boundary changes are −0.28 and −0.52 pp, and offset-16
filtered-rank correlation lies between its same-type neighbors rather than peaking.
There is no evidence in this capture that the repeated regions explain the readout
advantage. This does not establish zero bias in the authors' BF16 results or statistical
equivalence across copies. Common-filter gains are +31.3/+27.8/+22.5 pp.

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

The email can communicate the architecture finding and the measured readout null now.
Matched-slot interventions address the remaining causal question: the existing bands
cross or occupy different portions of the repeated regions, and this work has not yet
compared aligned slots with appropriate perturbation-dose controls.
