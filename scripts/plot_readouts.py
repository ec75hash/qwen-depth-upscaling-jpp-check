#!/usr/bin/env python3
"""Plot the October 12 readout result from the released rank reanalysis."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/jpp_readouts/reanalysis.json")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    r = json.loads(args.data.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    ax = axes[0]
    for lo, hi, color in [(8, 23, "#e2f0ee"), (24, 39, "#f3e8d8"), (40, 55, "#e8e4f4")]:
        ax.axvspan(lo-.5, hi+.5, color=color)
    for key, label, color, style in [("jpp|filt", "J++ filtered", "#176b87", "-"),
                                     ("jlens|full", "J-lens full vocabulary", "#777777", "--"),
                                     ("jlens|filt", "J-lens filtered", "#b06222", ":")]:
        ax.plot(range(63), 100*np.array(r["per_layer"][key]), style, label=label, color=color, linewidth=1.6)
    for boundary in (23.5, 39.5): ax.axvline(boundary, color="#777777", linewidth=.7)
    ax.set(title="Readouts across all 63 lens layers", xlabel="Decoder block output (zero-based)",
           ylabel="Macro recall@10 (%)", ylim=(0, 50))
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(.02, .02, "J++ at internal boundaries: −0.28 / −0.52 pp", transform=ax.transAxes,
            fontsize=8.5, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 2})

    ax = axes[1]; x = np.arange(3)
    names = ["copyA_8_23", "copyB_24_39", "copyC_40_55"]
    published = [100*r["layer_sets"][n]["gain_published_convention"] for n in names]
    common = [100*r["layer_sets"][n]["gain_common_filter"] for n in names]
    for offset, values, label, color in [(-.18, published, "J++ filtered − J-lens full", "#176b87"),
                                        (.18, common, "Both filtered", "#b06222")]:
        bars = ax.bar(x+offset, values, width=.34, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x, ["A: 8–23", "B: 24–39", "C: 40–55"])
    ax.set(title="Readout advantage in each repeated region", xlabel="Inferred repeated region",
           ylabel="J++ − J-lens (percentage points)", ylim=(0, 43))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax = axes[2]
    for period, offsets in [("16", [12, 16, 20]), ("32", [28, 32, 36])]:
        stats = r["aligned_copy_similarity"]["jpp"][period]
        vals = [stats[f"k={k}"] for k in offsets]
        ax.plot(offsets, vals, ".-", color="#176b87", linewidth=1.4, markersize=8)
        for k, value in zip(offsets, vals): ax.annotate(f"{value:.3f}", (k, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        ax.axvline(int(period), color="#e87926", linewidth=.8, alpha=.6)
    ax.set(title="No distinct period-16/32 readout peak", xlabel="Offset (same-type controls on either side)",
           ylabel="Mean Spearman correlation of filtered ranks", ylim=(.35, .8))
    ax.set_xticks([12,16,20,28,32,36])
    fig.suptitle("Repeated ancestry in weights, without a corresponding J++ readout signature", fontsize=14)
    fig.supxlabel("October 12 capture: 329 locally held-out items · MLX 4-bit · different lens-fitting budgets · saved-rank reanalysis", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.out / f"jpp_readout_check.{extension}", dpi=180)
    plt.close(fig)
    print("Wrote J++ readout figure")


if __name__ == "__main__":
    main()
