#!/usr/bin/env python3
"""Verify checksums, tensor namespace, data coverage, and the release's architecture claims."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    checksums = json.loads((root / "checksums.json").read_text())
    for name, expected in checksums.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"FAIL checksum: {name}")
    summaries = json.loads((root / "data/summary.json").read_text())["analyses"]
    provenance = json.loads((root / "data/provenance.json").read_text())
    lock = json.loads((root / "models.lock.json").read_text())
    if provenance["models_lock_sha256"] != hashlib.sha256((root / "models.lock.json").read_bytes()).hexdigest():
        raise SystemExit("FAIL source lock hash")
    if provenance["scanner_sha256"] != hashlib.sha256((root / "scripts/scan_weights.py").read_bytes()).hexdigest():
        raise SystemExit("FAIL scanner hash")
    models = {m["repo_id"]: m for m in lock["models"]}
    for model in provenance["models"]:
        spec = models[model["repo_id"]]
        for tensor in model["selected_tensors"]:
            if not tensor["tensor_name"].startswith(spec["decoder_prefix"]):
                raise SystemExit("FAIL auxiliary tensor selected")
    lookup = {(a["model_slug"], a["tensor"]): a for a in summaries}
    conv = lookup[("qwen36-27b", "linear_attn.conv1d.weight")]
    key = lookup[("qwen36-27b", "self_attn.k_proj.weight")]
    assert conv["depth"] == 64 and len(conv["tensor_layers"]) == 48
    assert key["depth"] == 64 and len(key["tensor_layers"]) == 16
    assert conv["top_offsets"][0]["offset"] == key["top_offsets"][0]["offset"] == 16
    assert conv["periods"]["16"]["n_pairs"] == 36
    assert conv["periods"]["32"]["n_pairs"] == 24
    assert key["periods"]["16"]["n_pairs"] == 12
    # Qualitative architecture checks fixed before this package's capture.
    assert conv["periods"]["16"]["inside_min"] > .5
    assert conv["periods"]["16"]["outside_mean"] < .05
    solar = lookup[("solar-10p7b", "self_attn.k_proj.weight")]
    assert solar["top_offsets"][0]["offset"] == 16
    assert solar["periods"]["16"]["inside_min"] > .5
    for tensor in ("linear_attn.conv1d.weight", "self_attn.k_proj.weight"):
        control = lookup[("qwen35-9b", tensor)]
        assert max(r["mean_cosine"] for r in control["top_offsets"]) < .05
    pairs = list(csv.DictReader((root / "data/pairs.csv").open()))
    for a in summaries:
        count = sum(r["model_slug"] == a["model_slug"] and r["tensor"] == a["tensor"] for r in pairs)
        n = len(a["tensor_layers"])
        assert count == n * (n - 1) // 2
    print(f"PASS: {len(checksums)} checksums, five tensor analyses, source provenance, decoder namespaces, "
          f"{len(pairs)} layer pairs, primary pattern and controls")


if __name__ == "__main__":
    main()
