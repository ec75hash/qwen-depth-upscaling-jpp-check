# Derived data and provenance

- `pairs.csv`: one unordered pair of decoder blocks per model and tensor family.
  Columns: model ID, model slug, pinned revision, tensor suffix, layer_a, layer_b,
  offset (= layer_b−layer_a), flattened weight cosine, and whether both endpoints
  fall inside the declared reporting region. This is the primary derived measurement.
- `offsets.csv`: mean cosine and number of pairs for each supported offset, with
  `rank_eligible` indicating the six-pair reporting cutoff. Pairless offsets are omitted.
- `summary.json`: full decoder/tensor coverage, period-16/32 all-pair and region-only
  means/counts/ranges, overall pair mean, and count-filtered top offsets.
- `provenance.json`: source revisions, selected tensor names, shapes, dtypes, exact byte
  ranges, payload/header SHA-256s, scanner/source-lock hashes, software versions, runtime.
- `../provenance/metadata/`: small upstream config and safetensors index files at the
  pinned revisions. They identify tensors and configuration; they contain no tensor payloads.
- `jpp_readouts/`: the separate October 12 readout evidence: 99,414 saved rank rows,
  source aggregates, sanitized capture provenance, and a regenerated summary. See its
  [data dictionary](jpp_readouts/README.md) for scoring and reproduction scope.

Rows within a checkpoint share weights and are not independent statistical trials.
There are no personal records or training examples in these files. `analysis` in the
Hugging Face card is a dataset split name for measured layer pairs, not a training split.
The separate `held_out` rank split refers to a locally reconstructed evaluation split.
