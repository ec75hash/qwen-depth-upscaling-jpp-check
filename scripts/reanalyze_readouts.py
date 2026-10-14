#!/usr/bin/env python3
"""Recompute block gains, layer curves, boundary deltas, and offset controls from saved ranks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LENSES = ("jpp", "jlens", "logit")
SETS = {"authors_7": [8, 16, 24, 32, 40, 48, 56], "all_0_62": list(range(63)),
        "copyA_8_23": list(range(8, 24)), "copyB_24_39": list(range(24, 40)),
        "copyC_40_55": list(range(40, 56))}


def average_ranks(x):
    order = np.argsort(x, kind="stable")
    _, start, count = np.unique(x[order], return_index=True, return_counts=True)
    ranked = np.empty(len(x), dtype=float)
    ranked[order] = np.repeat(start + (count - 1) / 2, count)
    return ranked


def spearman(x, y):
    # Log ranks are monotonic in ranks: tied-rank Spearman is unchanged by log().
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def analyze(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert {r["split"] for r in rows} == {"held-out"}
    units = sorted({(r["eval"], r["item"], r["intermediate"]) for r in rows})
    items = sorted({u[:2] for u in units})
    u_index, i_index = {u: n for n, u in enumerate(units)}, {u: n for n, u in enumerate(items)}
    unit_item = np.array([i_index[u[:2]] for u in units])
    item_eval = np.array([u[0] for u in items])
    tasks = sorted(set(item_eval))
    assert len(items) == 329 and len(tasks) == 5
    ranks = {(lens, mode): np.full((63, len(units)), -1, dtype=np.int32)
             for lens in LENSES for mode in ("full", "filt")}
    for r in rows:
        u = u_index[(r["eval"], r["item"], r["intermediate"])]; layer = int(r["layer"])
        for mode, col in (("full", "rank_full"), ("filt", "rank_filtered")):
            assert ranks[r["lens"], mode][layer, u] == -1, "Duplicate observation"
            ranks[r["lens"], mode][layer, u] = int(r[col])
    assert all((v > 0).all() for v in ranks.values()), "Missing or invalid ranks"
    counts = np.bincount(unit_item)

    def macro(minrank):
        item_hit = np.bincount(unit_item, weights=minrank <= 10) / counts
        return float(np.mean([item_hit[item_eval == task].mean() for task in tasks]))

    block = {}
    for name, layers in SETS.items():
        rates = {f"{lens}|{mode}": macro(ranks[lens, mode][layers].min(axis=0))
                 for lens in LENSES for mode in ("full", "filt")}
        rates["gain_published_convention"] = rates["jpp|filt"] - rates["jlens|full"]
        rates["gain_common_filter"] = rates["jpp|filt"] - rates["jlens|filt"]
        block[name] = rates
    curves = {f"{lens}|{mode}": [macro(row) for row in ranks[lens, mode]]
              for lens in LENSES for mode in ("full", "filt")}
    boundaries = {lens: {f"{l}->{l+1}": curves[f"{lens}|filt"][l+1] - curves[f"{lens}|filt"][l]
                         for l in range(3, 60, 4)} for lens in LENSES}
    aligned = {}
    for lens in LENSES:
        aligned[lens] = {}
        for period, offsets in ((16, (12, 16, 20)), (32, (28, 32, 36))):
            anchors = [l for l in range(63) if 8 <= l and l + max(offsets) <= 55]
            means = {f"k={k}": float(np.mean([spearman(ranks[lens, "filt"][l],
                                                        ranks[lens, "filt"][l+k]) for l in anchors]))
                     for k in offsets}
            means["excess"] = means[f"k={period}"] - (means[f"k={offsets[0]}"] + means[f"k={offsets[2]}"]) / 2
            means["n_anchors"] = len(anchors)
            aligned[lens][str(period)] = means
    return {"n_items": len(items), "n_units": len(units), "n_layers": 63,
            "layer_sets": block, "per_layer": curves, "boundary_deltas": boundaries,
            "aligned_copy_similarity": aligned}


def check_original(result, source):
    for layer_set, rates in result["layer_sets"].items():
        for lens in LENSES:
            for mode in ("full", "filt"):
                expected = source["pass_at_k"][f"{lens}|{mode}|{layer_set}|k10"]["macro5"]
                assert abs(rates[f"{lens}|{mode}"] - expected) < 1e-12
    for key, curve in result["per_layer"].items():
        np.testing.assert_allclose(curve, source["per_layer"][key], atol=1e-12, rtol=0)
    for lens in LENSES:
        for period in ("16", "32"):
            expected = source["aligned_copy_similarity_spearman_logrank"][lens][period]["inside_block"]
            for name, value in result["aligned_copy_similarity"][lens][period].items():
                assert abs(value - expected[name]) <= .000051, (lens, period, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/jpp_readouts")
    parser.add_argument("--out", type=Path, default=ROOT / "data/jpp_readouts/reanalysis.json")
    args = parser.parse_args()
    result = analyze(args.data / "held_out_ranks.csv")
    check_original(result, json.loads((args.data / "capture_summary.json").read_text()))
    provenance = json.loads((args.data / "provenance.json").read_text())
    assert hashlib.sha256((args.data / "held_out_ranks.csv").read_bytes()).hexdigest() == provenance["exported_ranks_sha256"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, rates in result["layer_sets"].items():
        print(name, f"gain={100*rates['gain_published_convention']:.2f} pp; common-filter gain={100*rates['gain_common_filter']:.2f} pp")
    print(f"PASS: {result['n_items']} items, {result['n_units']} units, {result['n_layers']} layers; "
          "block scores, layer curves and offset correlations reproduce October 12 aggregates")


if __name__ == "__main__":
    main()
