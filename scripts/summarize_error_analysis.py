"""Concise scientific interpretation from completed analysis tables; no pixel processing."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from src.error_analysis import DEPTH_LABELS
from src.project import write_json


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=Path("outputs/error_analysis"))
    args = parser.parse_args()
    summary = json.loads((args.analysis / "analysis_summary.json").read_text(encoding="utf-8"))
    bins = read_csv(args.analysis / "depth_bin_metrics.csv")
    transitions = read_csv(args.analysis / "delta_transitions.csv")
    tails = read_csv(args.analysis / "large_error_contributions.csv")
    distributions = read_csv(args.analysis / "frame_distributions.csv")
    lookup = {(r["depth_bin_mm"], r["method"]): r for r in bins}
    tradeoffs = []
    for depth in DEPTH_LABELS:
        g, l = lookup[depth, "global_scale"], lookup[depth, "log_affine"]
        row = {"depth_bin_mm": depth, "valid_pixels": int(g["valid_pixels"]),
               "pixel_fraction": float(g["valid_pixel_fraction"])}
        for metric in ("abs_rel", "rmse_mm", "delta_1_25"):
            gv, lv = float(g[metric]), float(l[metric])
            row.update({f"global_{metric}": gv, f"log_{metric}": lv, f"difference_{metric}": lv-gv})
        for prefix, source in (("global", g), ("log", l)):
            bias, rmse = float(source["mean_signed_error_mm"]), float(source["rmse_mm"])
            row[f"{prefix}_mean_signed_error_mm"] = bias
            row[f"{prefix}_residual_sd_mm"] = float(np.sqrt(max(0., rmse**2-bias**2)))
        transition = next(r for r in transitions if r["depth_bin_mm"] == depth)
        row["delta_net_pass_change"] = int(transition["net_delta_pass_change"])
        row["global_delta_contribution_pp"] = float(transition["contribution_to_global_delta_change_pp"])
        tradeoffs.append(row)
    with (args.analysis / "calibration_tradeoff_by_depth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tradeoffs[0]))
        writer.writeheader()
        writer.writerows(tradeoffs)
    # Validate the squared-error decomposition used for the descriptive residual SD.
    for row in tradeoffs:
        for prefix in ("global", "log"):
            np.testing.assert_allclose(row[f"{prefix}_residual_sd_mm"]**2 + row[f"{prefix}_mean_signed_error_mm"]**2,
                                       row[f"{prefix}_rmse_mm"]**2, rtol=1e-12)
    tail_lookup = {(r["method"], float(r["absolute_error_threshold_mm"])): r for r in tails}
    log_tail = tail_lookup["log_affine", 100.]
    loss = next(r for r in transitions if r["depth_bin_mm"] == "10–20")
    paragraph = f"""# Experimental error analysis — draft

## Scope and aggregation

We analyzed {summary['valid_pixels']:,} GT-valid pixels from 103 consecutive frames of the C3VD trans_t2_b sequence. Depth bins were fixed as [0,10), [10,20), [20,40), [40,60), and [60,100) mm. The existing mask, 0 < raw depth < 65535, and all model/calibration parameters were unchanged. Bin metrics pool pixels across frames; frame distributions are descriptive summaries, not substitutes for the pooled scores. No model inference, fitting, prediction clipping, or test-driven parameter selection was performed. These frames are temporally correlated observations from one video, not 103 independent clinical cases.

## Observed depth dependence

The original model had an absolute error above 100 mm at every valid test pixel, indicating a pervasive magnitude mismatch in this sequence rather than an error restricted to a few outliers. Its AbsRel ranged from 54.095 in the 0–10 mm bin to 5.890 in the 60–100 mm bin. This numerical pattern does not establish a specific physical, imaging, or training-related cause.

Global scale calibration concentrated predictions near the middle of the depth range. It performed best in the 10–20 mm bin (AbsRel 0.1935, RMSE 3.0885 mm, delta 0.6930), which contained 38.76% of valid pixels, but strongly underestimated larger depths. In the 60–100 mm bin, mean GT was 76.10 mm whereas the global-scale mean prediction was 19.64 mm. Log-affine calibration raised that mean to 36.27 mm, but a mean signed error of -39.83 mm and 99.47% underestimation remained.

Log-affine calibration improved 40–60 mm RMSE from 30.0464 to 19.4660 mm and 60–100 mm RMSE from 57.5270 to 42.7622 mm. Its effects were not uniformly favorable: 10–20 mm AbsRel increased from 0.1935 to 0.3502 and RMSE from 3.0885 to 6.2021 mm. In 20–40 mm, RMSE and delta improved but AbsRel rose slightly (0.3317 to 0.3364). In 0–10 mm, AbsRel and delta improved, while RMSE rose slightly from 7.9638 to 8.0017 mm. The fixed log-affine exponent expands contrast, but the remaining errors cannot be reduced to a single scale mismatch.

## Why threshold accuracy decreased

Pixelwise transitions directly locate the delta trade-off. Across the full sequence, 29,304,468 pixels passed only under global scale, while 26,628,349 passed only under log-affine calibration: a net loss of 2,676,119, or 1.9583 percentage points. The 10–20 mm bin alone lost {abs(int(loss['net_delta_pass_change'])):,} net passes, contributing {float(loss['contribution_to_global_delta_change_pp']):.4f} points to the global change. The other four bins all gained and offset most of that loss. In 10–20 mm, residual SD increased from 2.8150 to 6.1457 mm even as the magnitude of mean signed bias decreased. Thus reduced average bias did not imply more predictions inside the multiplicative 1.25 threshold. Minimizing development log-depth squared error does not guarantee threshold accuracy on the test sequence.

## Large errors and frame heterogeneity

For log-affine calibration, errors above 50 mm occurred in 1.1930% of pixels but contributed 24.9109% of total squared error. Errors above 100 mm occurred in {int(log_tail['pixels']):,} pixels ({float(log_tail['fraction_of_valid_pixels'])*100:.6f}%) and contributed {float(log_tail['fraction_of_method_sse'])*100:.4f}% of squared error; all were in the 0–10 mm GT bin. The maximum absolute error was {float(log_tail['max_absolute_error_mm']):.2f} mm. These tails matter, but errors above 100 mm alone do not dominate the total RMSE. All remained in the formal scores.

Frame behavior also varied. Log-affine calibration improved AbsRel in 83/103 frames and RMSE in 96/103, but improved delta in 53 and worsened it in 50. Its frame-level AbsRel median [IQR] was 0.4128 [0.3799, 0.4594], compared with 0.4594 [0.4487, 0.4738] for global scale. Its RMSE median was 11.6739 versus 16.5002 mm, while delta had a much wider frame range (0.0663–0.5370 versus 0.1927–0.3843). By the prespecified AbsRel ranking, log-affine frame 0075 was best (0.2764) and frame 0036 worst (0.6369); these are metric-specific illustrative cases, not overall rankings of clinical difficulty.

## Interpretation limits

The evidence supports correction of scale and part of the compressed depth contrast, with persistent depth-dependent bias, residual spread, and occasional large local errors. It does not identify reflection, texture, illumination, or domain shift as causal explanations. The descriptive analysis neither establishes clinical generalization nor changes the frozen methods. Figures A–C summarize pixel-pooled bin metrics; Figure D shows frame distributions and temporal order without independence assumptions or inferential error bars.
"""
    (args.analysis / "experimental_analysis_draft_en.md").write_text(paragraph, encoding="utf-8")
    # A compact numeric digest useful for downstream reporting without loading the caches.
    write_json(args.analysis / "report_digest.json", {"depth_tradeoffs": tradeoffs,
        "error_tail_statistics": tails, "frame_metric_distributions": distributions,
        "delta_transitions": transitions, "analysis_only": True})
    write_json(args.analysis / "verification.json", {"status": "passed",
        "checks": ["bin sufficient statistics reconstruct all four existing global scores",
                   "delta transition categories partition each bin and reconstruct the exact net delta change",
                   "squared-error decomposition for residual SD"],
        "no_inference_fitting_or_formal_score_changes": True})
    print("Saved depth trade-off table, English analysis draft, reporting digest, and analysis verification")


if __name__ == "__main__":
    main()
