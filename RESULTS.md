# Numerical results

Regenerated from `data/summary.json` by `scripts/make_report.py`.

Cosines use flattened decoder-only tensors. Means at each offset include every valid pair;
region means are reported separately. Offset rankings require at least six pairs.

| Checkpoint | Tensor | Offset | All-pair mean | Pairs | Region mean | Region pairs |
|---|---|---:|---:|---:|---:|---:|
| Qwen/Qwen3.6-27B | `linear_attn.conv1d.weight` | 16 | 0.572769 | 36 | 0.858496 | 24 |
| Qwen/Qwen3.6-27B | `linear_attn.conv1d.weight` | 32 | 0.399886 | 24 | 0.799515 | 12 |
| Qwen/Qwen3.6-27B | `self_attn.k_proj.weight` | 16 | 0.206127 | 12 | 0.309258 | 8 |
| Qwen/Qwen3.6-27B | `self_attn.k_proj.weight` | 32 | 0.095644 | 8 | 0.191269 | 4 |
| Qwen/Qwen3.5-9B | `linear_attn.conv1d.weight` | 16 | -0.000381 | 12 | -0.000381 | 12 |
| Qwen/Qwen3.5-9B | `linear_attn.conv1d.weight` | 32 | — | 0 | — | 0 |
| Qwen/Qwen3.5-9B | `self_attn.k_proj.weight` | 16 | 0.000060 | 4 | 0.000060 | 4 |
| Qwen/Qwen3.5-9B | `self_attn.k_proj.weight` | 32 | — | 0 | — | 0 |
| upstage/SOLAR-10.7B-v1.0 | `self_attn.k_proj.weight` | 16 | 0.435458 | 32 | 0.870722 | 16 |
| upstage/SOLAR-10.7B-v1.0 | `self_attn.k_proj.weight` | 32 | 0.000088 | 16 | — | 0 |

For the negative control, ‘region’ is the whole decoder; it is not a proposed copied block.

## Strongest offsets

- **Qwen/Qwen3.6-27B**, `linear_attn.conv1d.weight`: k=16: 0.572769 (n=36); k=32: 0.399886 (n=24); k=39: 0.003553 (n=12); k=53: 0.003493 (n=6).
- **Qwen/Qwen3.6-27B**, `self_attn.k_proj.weight`: k=16: 0.206127 (n=12); k=32: 0.095644 (n=8); k=4: 0.000483 (n=15); k=20: 0.000450 (n=11).
- **Qwen/Qwen3.5-9B**, `linear_attn.conv1d.weight`: k=19: 0.007049 (n=6); k=6: 0.004049 (n=13); k=21: 0.002522 (n=6); k=9: 0.000695 (n=12).
- **Qwen/Qwen3.5-9B**, `self_attn.k_proj.weight`: k=8: 0.000854 (n=6); k=4: 0.000530 (n=7).
- **upstage/SOLAR-10.7B-v1.0**, `self_attn.k_proj.weight`: k=16: 0.435458 (n=32); k=3: 0.000497 (n=45); k=19: 0.000453 (n=29); k=13: 0.000333 (n=35).

## Interpretation

Qwen3.6-27B has offset-16 and offset-32 weight-similarity peaks, with a sharp offset-16
plateau across the proposed repeated region. Both DeltaNet convolution and attention key
projections carry the pattern. The disclosed-duplication SOLAR control recovers its known
offset-16 block. The Qwen3.5-9B control lacks comparable peaks in the tested tensors.

This supports a shared block-ancestry interpretation. It does not establish the unpublished
construction recipe, functional interchangeability, or compensation between copies.

All supported offsets, including those with fewer than six pairs, are retained in the CSV.
