"""Targeted validation of the new log-affine results, without old full audits."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from src.c3vd_dataset import C3VDDataset
from src.evaluation import evaluation_mask, file_sha256
from src.project import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", type=Path, default=Path("outputs/results"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/calibration"))
    parser.add_argument("--output", type=Path, default=Path("outputs/contrast_calibration"))
    args = parser.parse_args()
    read = lambda name: json.loads((args.output / name).read_text(encoding="utf-8"))
    fit, development, test = read("log_affine_fit.json"), read("development_summary.json"), read("test_summary.json")
    fingerprint = file_sha256(args.output / "log_affine_fit.json")
    if fingerprint != development["log_affine_fit_sha256"] or fingerprint != test["log_affine_fit_sha256"]:
        raise ValueError("Frozen parameters differ across development/test")
    for path, expected in test["prior_result_fingerprints"].items():
        if file_sha256(path) != expected:
            raise ValueError("Previous result was modified")
    statistics = fit["statistics"]
    np.testing.assert_allclose([fit["b"], fit["a"]],
        [statistics["centered_sum_xy"] / statistics["centered_sum_xx"],
         statistics["mean_log_gt"] - fit["b"] * statistics["mean_log_prediction"]], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(development["direct_log_sse"], statistics["fitted_log_sse"], rtol=1e-12)
    config = load_config(args.config)
    checked_rows = 0
    for name, summary, frames, count, directory in (
        ("development_per_frame.csv", development, 61, 183, args.previous / "development" / "predictions_mm"),
        ("test_per_frame.csv", test, 103, 412, args.reference / "predictions_mm")):
        with (args.output / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != count:
            raise ValueError("Incorrect number of new experiment per-frame rows")
        checked_rows += len(rows)
        for method, result in summary["results"].items():
            relevant = [row for row in rows if row["method"] == method]
            if len(relevant) != frames or [row["frame_id"] for row in relevant] != summary["frame_ids"]:
                raise ValueError("Methods use different frame sets")
            n = sum(int(row["valid_pixels"]) for row in relevant)
            if n != result["valid_pixels"]:
                raise ValueError("Per-frame pixel count mismatch")
            pooled = [sum(float(row["abs_rel"]) * int(row["valid_pixels"]) for row in relevant) / n,
                np.sqrt(sum(float(row["rmse_mm"]) ** 2 * int(row["valid_pixels"]) for row in relevant) / n),
                sum(float(row["delta_1_25"]) * int(row["valid_pixels"]) for row in relevant) / n]
            np.testing.assert_allclose(pooled, [result[k] for k in ("abs_rel", "rmse_mm", "delta_1_25")],
                                       rtol=1e-12, atol=1e-12)
        dataset = C3VDDataset(config["data_root"], summary["sequence"])
        for index in (0, frames//2, frames-1):
            raw, gt, _ = dataset.load_depth(index)
            pair = dataset.pairs[index]
            original = np.load(directory / f"{pair.frame_id}_prediction_mm.npy", allow_pickle=False)
            mask = evaluation_mask(raw)
            prediction = np.exp(fit["a"] + fit["b"] * np.log(original[mask].astype(np.float64)))
            gt_valid = gt[mask].astype(np.float64)
            direct = [np.mean(np.abs(prediction-gt_valid) / gt_valid),
                      np.sqrt(np.mean((prediction-gt_valid) ** 2)),
                      np.mean(np.maximum(prediction/gt_valid, gt_valid/prediction) < 1.25)]
            row = next(row for row in rows if row["method"] == "log_affine" and row["frame_id"] == pair.frame_id)
            np.testing.assert_allclose(direct, [float(row[k]) for k in ("abs_rel", "rmse_mm", "delta_1_25")],
                                       rtol=1e-12, atol=1e-12)
    write_json(args.output / "verification.json", {"status": "passed", "per_frame_rows_checked": checked_rows,
        "checks": ["one frozen parameter file used across both splits", "prior result metadata unchanged",
                   "recorded streaming OLS solution and independent direct residual SSE",
                   "new per-frame metric pooling", "direct formulas on first/middle/last frames of both splits"],
        "model_inference_performed": False, "previous_full_prediction_audits_repeated": False})
    print("PASS: frozen log-affine fit, 595 per-frame rows, pooled metrics and fixed-sample direct formulas")


if __name__ == "__main__":
    main()
