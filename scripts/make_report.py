#!/usr/bin/env python3
"""Regenerate the note's numerical tables and standalone figures from released CSV/JSON."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    parser.add_argument("--report", type=Path, default=ROOT / "RESULTS.md")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    analyses = json.loads((args.data / "summary.json").read_text())["analyses"]
    pairs = list(csv.DictReader((args.data / "pairs.csv").open()))
    offsets = list(csv.DictReader((args.data / "offsets.csv").open()))
    lookup = {(a["model_slug"], a["tensor"]): a for a in analyses}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})

    panels = [
        ("qwen36-27b", "linear_attn.conv1d.weight", "Qwen3.6-27B · DeltaNet convolution"),
        ("qwen36-27b", "self_attn.k_proj.weight", "Qwen3.6-27B · attention key projection"),
        ("solar-10p7b", "self_attn.k_proj.weight", "SOLAR-10.7B · disclosed duplication control"),
        ("qwen35-9b", "linear_attn.conv1d.weight", "Qwen3.5-9B · negative control"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#eeeeee")
    for ax, (slug, tensor, title) in zip(axes.flat, panels):
        a = lookup[(slug, tensor)]
        matrix = np.full((a["depth"], a["depth"]), np.nan)
        for row in pairs:
            if row["model_slug"] == slug and row["tensor"] == tensor:
                left, right = int(row["layer_a"]), int(row["layer_b"])
                matrix[left, right] = matrix[right, left] = float(row["cosine"])
        plot = ax.imshow(matrix, vmin=-.05, vmax=1, cmap=cmap, origin="lower", interpolation="nearest")
        if slug != "qwen35-9b":
            lo, hi = a["region"]
            for boundary in range(lo, hi + 2, a["period"]):
                ax.axhline(boundary - .5, color="#e87926", linewidth=.7, alpha=.85)
                ax.axvline(boundary - .5, color="#e87926", linewidth=.7, alpha=.85)
        ax.set(title=title, xlabel="Decoder block (zero-based)", ylabel="Decoder block (zero-based)")
        ax.set_xticks(range(0, a["depth"], 8)); ax.set_yticks(range(0, a["depth"], 8))
    fig.colorbar(plot, ax=axes, shrink=.75, label="Flattened weight cosine")
    fig.suptitle("Repeated block structure in decoder weights", fontsize=15)
    fig.supxlabel("Gray: tensor absent at that block pair, or omitted self-pair. Orange: inferred region boundaries.", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.out / f"weight_similarity.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row_id, tensor in enumerate(("linear_attn.conv1d.weight", "self_attn.k_proj.weight")):
        short = "DeltaNet convolution" if row_id == 0 else "Attention key projection"
        left, right = axes[row_id]
        for slug, label, color in (("qwen36-27b", "Qwen3.6-27B", "#176b87"),
                                   ("qwen35-9b", "Qwen3.5-9B control", "#8b8b8b")):
            selected = [r for r in offsets if r["model_slug"] == slug and r["tensor"] == tensor
                        and r["rank_eligible"] == "True"]
            left.plot([int(r["offset"]) for r in selected], [float(r["mean_cosine"]) for r in selected],
                      ".-", markersize=3, label=label, color=color, linewidth=1)
        for k in (16, 32):
            left.axvline(k, color="#e87926", alpha=.5, linewidth=.9)
        left.set(title=f"{short}: offset spectrum", xlabel="Offset in original decoder block IDs",
                 ylabel="Mean cosine (all valid pairs)", ylim=(-.05, 1))
        left.legend(frameon=False, fontsize=9)
        profile = [r for r in pairs if r["model_slug"] == "qwen36-27b" and r["tensor"] == tensor
                   and int(r["offset"]) == 16]
        x = [int(r["layer_a"]) for r in profile]
        y = [float(r["cosine"]) for r in profile]
        right.axvspan(7.5, 39.5, color="#e87926", alpha=.12, label="Both blocks inside inferred repeated region")
        right.plot(x, y, ".-", color="#176b87", linewidth=1)
        right.set(title=f"{short}: block ℓ versus ℓ + 16", xlabel="Source decoder block ℓ",
                  ylabel="Pair cosine", ylim=(-.05, 1), xlim=(-1, 48))
    fig.suptitle("Offset-16/32 peaks and localized offset-16 profiles", fontsize=15)
    fig.supxlabel("Spectra show offsets with ≥6 pairs; all supported offsets remain in data/offsets.csv.", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.out / f"offsets_and_boundaries.{extension}", dpi=180)
    plt.close(fig)

    text = ["# Numerical results", "", "Regenerated from `data/summary.json` by `scripts/make_report.py`.", "",
            "Cosines use flattened decoder-only tensors. Means at each offset include every valid pair;",
            "region means are reported separately. Offset rankings require at least six pairs.", "",
            "| Checkpoint | Tensor | Offset | All-pair mean | Pairs | Region mean | Region pairs |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for a in analyses:
        for offset, p in a["periods"].items():
            fmt = lambda x: "—" if x is None else f"{x:.6f}"
            text.append(f"| {a['model']} | `{a['tensor']}` | {offset} | {fmt(p['all_valid_mean'])} | "
                        f"{p['n_pairs']} | {fmt(p['inside_mean'])} | {p['n_inside']} |")
    text += ["", "For the negative control, ‘region’ is the whole decoder; it is not a proposed copied block.", "",
             "## Strongest offsets", ""]
    for a in analyses:
        tops = "; ".join(f"k={r['offset']}: {r['mean_cosine']:.6f} (n={r['n_pairs']})" for r in a["top_offsets"])
        text += [f"- **{a['model']}**, `{a['tensor']}`: {tops}."]
    text += ["", "## Interpretation", "",
             "Qwen3.6-27B has offset-16 and offset-32 weight-similarity peaks, with a sharp offset-16",
             "plateau across the proposed repeated region. Both DeltaNet convolution and attention key",
             "projections carry the pattern. The disclosed-duplication SOLAR control recovers its known",
             "offset-16 block. The Qwen3.5-9B control lacks comparable peaks in the tested tensors.", "",
             "This supports a shared block-ancestry interpretation. It does not establish the unpublished",
             "construction recipe, functional interchangeability, or compensation between copies.", "",
             "All supported offsets, including those with fewer than six pairs, are retained in the CSV.", ""]
    args.report.write_text("\n".join(text))
    print(f"Wrote figures and {args.report}")


if __name__ == "__main__":
    main()
