"""Four publication-style descriptive figures, from saved analysis tables only."""

import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.error_analysis import DEPTH_LABELS
from src.project import write_json

METHODS = ("constant_median", "original_depth_anything_v2", "global_scale", "log_affine")
LABELS = ("Constant median", "Original DAV2", "Global scale", "Log-affine")
COLORS = ("#777777", "#D55E00", "#0072B2", "#009E73")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=Path("outputs/error_analysis"))
    args = parser.parse_args()
    summary = json.loads((args.analysis / "analysis_summary.json").read_text(encoding="utf-8"))
    bins = read_csv(args.analysis / "depth_bin_metrics.csv")
    frames = read_csv(args.analysis / "frame_metrics.csv")
    output = args.analysis / "figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure_records = []
    for letter, metric, title, ylabel, logarithmic in (
        ("A", "abs_rel", "Depth-dependent absolute relative error", "AbsRel (dimensionless; logarithmic axis)", True),
        ("B", "rmse_mm", "Depth-dependent root mean squared error", "RMSE (mm; logarithmic axis)", True),
        ("C", "delta_1_25", "Depth-dependent threshold accuracy", "Fraction with max(pred/GT, GT/pred) < 1.25", False)):
        fig, axis = plt.subplots(figsize=(9.2, 5.6))
        x = np.arange(len(DEPTH_LABELS))
        for method, label, color in zip(METHODS, LABELS, COLORS):
            values = [float(next(r for r in bins if r["method"] == method and r["depth_bin_mm"] == depth)[metric])
                      for depth in DEPTH_LABELS]
            axis.plot(x, values, marker="o", markersize=6, linewidth=1.8, label=label, color=color)
        if logarithmic:
            axis.set_yscale("log")
        else:
            axis.set_ylim(-.025, 1.025)
        fractions = [summary["depth_bin_pixel_counts"][depth] / summary["valid_pixels"] for depth in DEPTH_LABELS]
        axis.set_xticks(x, [f"{depth}\n{fraction:.1%} of pixels" for depth, fraction in zip(DEPTH_LABELS, fractions)])
        axis.set_xlabel("Ground-truth depth bin (mm; left-closed, right-open)", labelpad=10)
        axis.set_ylabel(ylabel)
        axis.set_title(f"{letter}. {title}", fontsize=14, pad=14)
        axis.grid(axis="y", which="major", alpha=.25)
        axis.legend(frameon=True, framealpha=.95, ncol=2, fontsize=9, loc="best")
        fig.text(.5, .025, "All GT-valid pixels pooled across 103 sequential test frames; mask: 0 < raw depth < 65535.\n"
                 "Pixel fractions describe the evaluation population; no frame averaging or prediction filtering.",
                 ha="center", va="bottom", fontsize=8.5)
        fig.tight_layout(rect=[0, .10, 1, 1])
        stem = f"Figure_{letter}_{metric}_by_depth"
        fig.savefig(output / f"{stem}.png", dpi=300)
        fig.savefig(output / f"{stem}.pdf")
        plt.close(fig)
        figure_records.append({"figure": letter, "file_stem": stem, "y_scale": "logarithmic" if logarithmic else "linear"})
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for row_index, (metric, label, logarithmic) in enumerate(
        (("abs_rel", "AbsRel", True), ("rmse_mm", "RMSE (mm)", True), ("delta_1_25", "δ < 1.25 (fraction)", False))):
        values_by_method = []
        for method, name, color in zip(METHODS, LABELS, COLORS):
            selected = sorted([r for r in frames if r["method"] == method], key=lambda r: int(r["frame_id"]))
            values = np.array([float(r[metric]) for r in selected])
            values_by_method.append(values)
            axes[row_index, 0].plot([int(r["frame_id"]) for r in selected], values, color=color, label=name, linewidth=1.3)
        boxes = axes[row_index, 1].boxplot(values_by_method, tick_labels=LABELS, patch_artist=True, widths=.55,
            medianprops={"color": "black", "linewidth": 1.5}, flierprops={"marker": ".", "markersize": 3})
        for patch, color in zip(boxes["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(.55)
        for column in (0, 1):
            axis = axes[row_index, column]
            axis.set_ylabel(label + (" — log axis" if logarithmic else ""))
            axis.grid(axis="y", alpha=.22)
            if logarithmic:
                axis.set_yscale("log")
            else:
                axis.set_ylim(-.02, .65)
        axes[row_index, 0].set_xlim(0, 102)
        axes[row_index, 1].tick_params(axis="x", labelsize=8)
    axes[0, 0].set_title("Temporal order (same continuous video)")
    axes[0, 1].set_title("Descriptive frame distributions (median and IQR)")
    axes[0, 0].legend(ncol=2, fontsize=8, loc="best")
    axes[2, 0].set_xlabel("Frame ID")
    axes[2, 1].set_xlabel("Method")
    fig.suptitle("D. Frame-level metrics across 103 sequential test frames", fontsize=15)
    fig.text(.5, .013, "These are correlated video frames, not 103 independent clinical cases. No confidence intervals or significance tests are implied.\n"
             "Boxplot whiskers extend to 1.5×IQR; points beyond whiskers remain in all reported statistics.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, .055, 1, .96])
    stem = "Figure_D_frame_metric_distributions"
    fig.savefig(output / f"{stem}.png", dpi=300)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)
    figure_records.append({"figure": "D", "file_stem": stem, "absrel_rmse_axis": "logarithmic", "delta_axis": "linear"})
    write_json(output / "figure_index.json", {"figures": figure_records, "formats": ["png_300_dpi", "pdf_vector"],
        "data_source": "saved error-analysis tables only", "predictions_or_results_modified": False})
    print(f"Saved Figures A–D as 300-dpi PNG and vector PDF to {output}")


if __name__ == "__main__":
    main()
