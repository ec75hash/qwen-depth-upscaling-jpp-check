#!/usr/bin/env python3
"""Export the public evaluation ranks and sanitized provenance from the October 12 capture."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--matrix-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data/jpp_readouts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source
    results = json.loads((source / "readout_results.json").read_text())
    cohort = json.loads((source / "cohort_manifest.json").read_text())
    capture = json.loads((source / "capture_27b_meta.json").read_text())
    matrices = json.loads(args.matrix_report.read_text())
    with (source / "ranks_long.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = [r for r in reader if r["split"] == "held-out"]
    assert all(None not in row for row in rows), "Malformed source CSV row"
    assert len({(r['eval'], r['item']) for r in rows}) == 329
    with (args.out / "held_out_ranks.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    # Preserve the source aggregate values as independent targets for reanalysis.
    keys = ["per_layer", "bootstrap", "boundary_deltas_pass10_filt", "aligned_copy_similarity_spearman_logrank"]
    summary = {key: results[key] for key in keys}
    summary["pass_at_k"] = results["pass_at_k"]["held_out_0"]
    summary["capture_date"] = "2026-10-12"
    provenance = {
        "capture_date": "2026-10-12", "forward_backend": "MLX 4-bit",
        "source_artifact_sha256": {name: sha(source / name) for name in
            ["ranks_long.csv", "readout_results.json", "cohort_manifest.json",
             "capture_27b_meta.json", "readout_score.py", "readout_capture.py"]},
        "matrix_report_sha256": sha(args.matrix_report),
        "exported_ranks_sha256": sha(args.out / "held_out_ranks.csv"),
        "authors_code_commit": cohort["jpp_lens_commit"],
        "model": cohort["hf_model_name"], "quantized_model_config_sha256": capture["model_sha_config"],
        "tokenizer_json_sha256": cohort["tokenizer_json_sha256"],
        "eval_file_sha256": cohort["eval_file_sha256"],
        "split": cohort["split"], "counts": cohort["counts"], "per_eval": cohort["per_eval"],
        "macro_evals": cohort["macro_evals"], "tokenization": cohort["tokenization"],
        "readout_rules": cohort["readout_rules"], "capture_convention": capture["convention"],
        "head_fidelity": {"argmax_agree": sum(x["argmax_agree"] for x in results["fidelity_pilot"]),
                          "n": len(results["fidelity_pilot"]),
                          "mean_top10_jaccard": sum(x["top10_jaccard"] for x in results["fidelity_pilot"]) / len(results["fidelity_pilot"]),
                          "max_abs_logit_diff": max(x["max_abs_logit_diff"] for x in results["fidelity_pilot"])},
        "lenses": {label: {"sha256": item["sha256"], "n_layers": item["n_layers"],
                            "d_model": item["d_model"]}
                   for label, item in matrices["lenses"].items()},
        "lens_labels": {"lens": "released J++ (64 fitting sequences)",
                        "control1": "Neuronpedia J-lens (1000 fitting prompts)"},
        "reproduction_scope": "Reanalysis of saved ranks; capture and lens fitting are not rerun by this release. "
                              "Quantized weights, lens files, activations and full original capture environment are not redistributed."
    }
    for name, value in [("capture_summary.json", summary), ("provenance.json", provenance)]:
        (args.out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(f"Exported {len(rows)} rank rows for 329 locally held-out items")


if __name__ == "__main__":
    main()
