"""Prepare report graphics and tables from saved summaries; no inference or fitting."""

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
METHODS = ("constant_median", "original_depth_anything_v2", "global_scale", "log_affine")
NAMES = ("Constant median", "Original DAV2", "Global scale", "Log-affine")
COLORS = ("#777777", "#D55E00", "#0072B2", "#009E73")
METRICS = ("abs_rel", "rmse_mm", "delta_1_25")
SOURCES = [
    "outputs/results/summary.json", "outputs/results/protocol.json", "outputs/results/baseline_fit.json",
    "outputs/calibration/scale_fit.json", "outputs/calibration/development_summary.json",
    "outputs/contrast_calibration/log_affine_fit.json",
    "outputs/contrast_calibration/development_summary.json", "outputs/contrast_calibration/test_summary.json",
    "outputs/error_analysis/depth_bin_metrics.csv", "outputs/error_analysis/frame_distributions.csv",
    "outputs/error_analysis/report_digest.json", "outputs/error_analysis/experimental_analysis_draft_en.md",
    "outputs/final_visualizations/visualization.json",
    "outputs/final_visualizations/0102_final_comparison.png",
    "outputs/error_analysis/figures/Figure_D_frame_metric_distributions.png",
]


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, name):
    fig.savefig(OUT / "figures" / (name + ".png"), dpi=300, facecolor="white")
    fig.savefig(OUT / "figures" / (name + ".svg"), facecolor="white")
    plt.close(fig)


def table(stem, headers, rows):
    with (OUT / "tables" / (stem + ".csv")).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    (OUT / "tables" / (stem + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def workflow():
    fig, ax = plt.subplots(figsize=(13.8, 7.6))
    ax.set(xlim=(0, 14), ylim=(0, 8))
    ax.axis("off")

    def box(x, y, w, h, text, color="#EDF3F8", fontsize=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.1",
                                   fc=color, ec="#657786", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.5)

    def arrow(start, end, rad=0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                   color="#405261", lw=1.3, connectionstyle=f"arc3,rad={rad}"))

    ax.text(.1, 7.7, "Development: fit global parameters using target-domain ground truth", fontsize=13, weight="bold")
    box(.1, 5.8, 2.35, 1.15, "trans_t1_a\n61 RGB + GT frames\n80,952,223 valid pixels")
    box(3.2, 5.8, 2.5, 1.15, "Frozen pretrained DAV2\nMetric Indoor Small\nCached original predictions")
    box(6.5, 5.65, 3.25, 1.45, "Development GT + predictions\nGlobal scale: median ratio s\nLog-affine: log-space OLS (a, b)", "#E5F3ED")
    box(6.5, 4.25, 3.25, .8, "Development GT only\nConstant median depth", "#F0F0F0")
    box(10.55, 5.1, 3.1, 1.4, "Freeze all parameters\nMedian = 17.25948 mm\ns = 0.03821440\na = -26.36792, b = 4.79152", "#FFF4DB")
    arrow((2.51, 6.37), (3.14, 6.37))
    ax.text(2.82, 6.55, "RGB", ha="center", fontsize=8)
    arrow((5.76, 6.37), (6.44, 6.37))
    # Separate GT connections make supervision explicit.
    ax.plot([1.3, 1.3, 6.15, 6.15], [5.74, 5.35, 5.35, 6.0], color="#405261", lw=1.1)
    arrow((6.15, 6.0), (6.44, 6.0))
    ax.plot([1.3, 1.3, 6.1], [5.35, 4.65, 4.65], color="#405261", lw=1.1)
    arrow((6.1, 4.65), (6.44, 4.65))
    arrow((9.81, 6.3), (10.49, 6.0))
    arrow((9.81, 4.65), (10.49, 5.35))
    ax.axhline(3.85, color="#B8C4CB", ls="--", lw=1)
    ax.text(.1, 3.5, "Test: apply fixed methods; ground truth is used only for evaluation", fontsize=13, weight="bold")
    box(.1, 1.65, 2.35, 1.15, "trans_t2_b\n103 RGB frames\nOne continuous video")
    box(3.2, 1.65, 2.5, 1.15, "Same frozen DAV2\nCached test predictions\nNo per-frame scaling")
    box(6.5, 1.4, 3.25, 1.6, "Four predictions\nConstant median baseline\nOriginal DAV2 (zero-shot)\nGlobal scale / Log-affine", "#E5F3ED")
    box(10.55, 1.4, 3.1, 1.6, "Pixel-pooled evaluation\nAbsRel / RMSE (mm) / δ\n136,655,314 valid pixels\n0 < raw GT < 65535")
    arrow((2.51, 2.22), (3.14, 2.22))
    arrow((5.76, 2.22), (6.44, 2.22))
    arrow((9.81, 2.22), (10.49, 2.22))
    # Routed fixed-parameter link avoids the test-evaluation box.
    ax.plot([12.1, 12.1, 8.1], [5.04, 3.25, 3.25], color="#405261", lw=1.3)
    arrow((8.1, 3.25), (8.1, 3.06))
    box(10.55, .15, 3.1, .65, "Test GT + frozen mask", "#F0F0F0")
    arrow((12.1, .86), (12.1, 1.34))
    ax.text(.1, .35, "Network weights unchanged. Both calibrations use development supervision.\nSchematic of completed experiments; report preparation reads saved results only.", fontsize=10, va="center")
    fig.tight_layout(pad=.6)
    save_figure(fig, "Figure_1_workflow")


def quantitative(results):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, metric, title, unit in zip(axes, METRICS,
            ("(a) Absolute relative error ↓", "(b) Root mean squared error ↓", "(c) Threshold accuracy ↑"),
            ("AbsRel (dimensionless; log axis)", "RMSE (mm; log axis)", "δ < 1.25 (fraction)")):
        values = [results[m][metric] for m in METHODS]
        ax.bar(np.arange(4), values, color=COLORS, width=.65, zorder=3)
        if metric != "delta_1_25":
            ax.set_yscale("log")
            ax.set_ylim((.2, 70) if metric == "abs_rel" else (8, 900))
        else:
            ax.set_ylim(0, .38)
        for i, value in enumerate(values):
            ax.text(i, value * 1.13 if metric != "delta_1_25" else value + .012,
                    f"{value:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(4), ("Constant\nmedian", "Original\nDAV2", "Global\nscale", "Log-\naffine"))
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.23, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Four-method comparison on the held-out test sequence", fontsize=14, y=.98)
    fig.text(.5, .015, "All 136,655,314 valid test pixels pooled; exact scores are in Table 1. Zero δ is shown at the baseline.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .92))
    save_figure(fig, "Figure_3_quantitative_comparison")


def depth_dependent(rows):
    depths = list(dict.fromkeys(r["depth_bin_mm"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9))
    for ax, metric, title, ylabel in zip(axes, METRICS,
            ("(a) AbsRel ↓", "(b) RMSE ↓", "(c) δ < 1.25 ↑"),
            ("AbsRel (dimensionless; log axis)", "RMSE (mm; log axis)", "Threshold accuracy (fraction)")):
        for method, name, color in zip(METHODS, NAMES, COLORS):
            values = [float(next(r for r in rows if r["method"] == method and r["depth_bin_mm"] == d)[metric]) for d in depths]
            ax.plot(range(5), values, "o-", color=color, label=name, lw=1.7, ms=5)
        if metric != "delta_1_25":
            ax.set_yscale("log")
        else:
            ax.set_ylim(-.025, .78)
        fractions = [float(next(r for r in rows if r["depth_bin_mm"] == d)["valid_pixel_fraction"]) for d in depths]
        ax.set_xticks(range(5), [f"{d}\n{f:.1%}" for d, f in zip(depths, fractions)], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("GT depth bin (mm) / share of valid pixels")
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=.23)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .92), ncol=4, frameon=False)
    fig.suptitle("Depth-dependent performance on the test sequence", fontsize=14, y=.99)
    fig.text(.5, .015, "Bins are left-closed and right-open. All valid pixels are retained; scores are pooled within each bin.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .85))
    save_figure(fig, "Figure_4_depth_dependent_error")


def distribution(quantiles):
    fig, ax = plt.subplots(figsize=(10, 4.4))
    entries = (("ground_truth_mm", "Ground truth", "#333333"),
               ("global_scale_mm", "Global scale", COLORS[2]),
               ("log_affine_mm", "Log-affine", COLORS[3]))
    for y, (key, label, color) in zip((2, 1, 0), entries):
        q = quantiles[key]
        ax.plot([q["p05"], q["p95"]], [y, y], color=color, lw=6, solid_capstyle="butt")
        ax.plot([q["p05"], q["p95"]], [y, y], "|", color=color, ms=17, mew=2)
        ax.plot(q["median"], y, "D", color=color, ms=8, markeredgecolor="white")
        # Outward alignment keeps the narrow global-scale endpoints readable.
        narrow = q["p95"] - q["p05"] < 12
        ax.text(q["p05"] - (.6 if narrow else 0), y + .18, f"P05 {q['p05']:.2f}",
                ha="right" if narrow else "center", fontsize=9)
        ax.text(q["p95"] + (.6 if narrow else 0), y + .18, f"P95 {q['p95']:.2f}",
                ha="left" if narrow else "center", fontsize=9)
        ax.text(q["median"], y - .25, f"Median {q['median']:.2f}", ha="center", fontsize=9)
    ax.set(yticks=[2, 1, 0], yticklabels=[e[1] for e in entries], xlim=(0, 100), ylim=(-.55, 2.6), xlabel="Depth (mm)")
    ax.set_title("Development depth distribution: P05–P95 interval and median", pad=15, fontsize=13)
    ax.grid(axis="x", alpha=.25)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    fig.text(.5, .02, "Saved exact quantiles of all 80,952,223 valid development pixels. Intervals are not confidence intervals or full ranges.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save_figure(fig, "Figure_5_development_depth_distribution")


def main():
    for sub in ("figures", "tables", "draft"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    dev = read_json("outputs/contrast_calibration/development_summary.json")
    test = read_json("outputs/contrast_calibration/test_summary.json")
    rows = read_csv("outputs/error_analysis/depth_bin_metrics.csv")
    table("Table_1_test_results", ["Method", "Target-domain supervision", "AbsRel (lower)", "RMSE (mm; lower)", "Delta < 1.25 (higher)"],
          [[name, supervision] + [f"{test['results'][m][k]:.8f}" for k in METRICS]
           for m, name, supervision in zip(METHODS, NAMES,
               ("Development GT median", "None: zero-shot pretrained inference", "Development GT + predictions: one global scale", "Development GT + predictions: global log-affine fit"))])
    table("Table_2_development_results", ["Method", "AbsRel", "RMSE (mm)", "Delta < 1.25"],
          [[name] + [f"{dev['results'][m][k]:.8f}" for k in METRICS]
           for m, name in zip(("original", "global_scale", "log_affine"), NAMES[1:])])
    table("Table_4_development_depth_quantiles", ["Depth source", "P05 (mm)", "Median (mm)", "P95 (mm)"],
          [[name] + [f"{dev['depth_quantiles'][m][k]:.8f}" for k in ("p05", "median", "p95")]
           for m, name in (("ground_truth_mm", "Ground truth"), ("global_scale_mm", "Global scale"), ("log_affine_mm", "Log-affine"))])
    bin_rows = []
    for depth in dict.fromkeys(r["depth_bin_mm"] for r in rows):
        g = next(r for r in rows if r["depth_bin_mm"] == depth and r["method"] == "global_scale")
        l = next(r for r in rows if r["depth_bin_mm"] == depth and r["method"] == "log_affine")
        bin_rows.append([depth, g["valid_pixels"], f"{float(g['valid_pixel_fraction'])*100:.4f}"] +
                        [f"{float(r[k]):.8f}" for k in METRICS for r in (g, l)])
    table("Table_3_calibration_by_depth", ["GT bin (mm)", "Valid pixels", "Share (%)", "Global AbsRel", "Log AbsRel", "Global RMSE (mm)", "Log RMSE (mm)", "Global delta", "Log delta"], bin_rows)
    # Copy existing analysis tables, retaining their complete numerical precision.
    for name in ("depth_bin_metrics.csv", "frame_distributions.csv", "representative_cases.csv", "delta_transitions.csv", "large_error_contributions.csv"):
        shutil.copy2(ROOT / "outputs/error_analysis" / name, OUT / "tables" / name)
    workflow()
    quantitative(test["results"])
    depth_dependent(rows)
    distribution(dev["depth_quantiles"])
    shutil.copy2(ROOT / "outputs/final_visualizations/0102_final_comparison.png", OUT / "figures/Figure_2_qualitative_0102.png")
    shutil.copy2(ROOT / "outputs/error_analysis/figures/Figure_D_frame_metric_distributions.png", OUT / "figures/Figure_S1_frame_metric_distributions.png")
    manifest = {"purpose": "Report assets from existing saved experimental outputs only",
                "model_inference": False, "parameter_fitting": False, "metric_reevaluation": False,
                "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
                "figures": {"1": "workflow schematic", "2": "byte-identical reuse of corrected 0102 visualization",
                            "3": "saved formal test scores", "4": "saved pixel-pooled depth-bin scores",
                            "5": "saved development quantiles; not a histogram or confidence interval",
                            "S1": "byte-identical reuse of existing frame-analysis figure"}}
    (OUT / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    draft = OUT / "draft/report_draft.md"
    if draft.exists():
        captions = re.findall(r"^\*\*Figure [1-5]\.\*\* .+$", draft.read_text(encoding="utf-8"), re.MULTILINE)
        captions.append("**Supplementary Figure S1.** Reused frame-level analysis across the 103 consecutive test frames. Left: metrics in temporal order. Right: descriptive boxplots with median, interquartile range, and 1.5-IQR whiskers. Points beyond whiskers remain in the scores. AbsRel and RMSE use log axes; delta is linear. The original analysis title 'D' is retained in this unchanged figure. Frames are correlated observations, not independent clinical cases; no confidence intervals or significance tests are implied.")
        (OUT / "draft/figure_captions.md").write_text("# Figure captions\n\n" + "\n\n".join(captions) + "\n", encoding="utf-8")
    print("Prepared 5 main figures, 1 reused supplementary figure, 4 Markdown/CSV tables and 5 precision-preserving analysis CSV copies.")


if __name__ == "__main__":
    main()
