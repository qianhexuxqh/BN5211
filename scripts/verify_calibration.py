"""Check calibration artifacts without inference, refitting or changing outputs."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from src.calibration import GlobalScaleCalibration
from src.c3vd_dataset import C3VDDataset
from src.evaluation import evaluation_mask, file_sha256
from src.project import load_config, write_json


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", type=Path, default=Path("outputs/results"))
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration"))
    args = parser.parse_args()
    fit = load_json(args.output / "scale_fit.json")
    dev = load_json(args.output / "development_summary.json")
    test = load_json(args.output / "test_summary.json")
    frozen_hash = file_sha256(args.output / "scale_fit.json")
    if frozen_hash != dev["scale_fit_sha256"] or frozen_hash != test["scale_fit_sha256"]:
        raise ValueError("Scale fingerprint mismatch")
    for filename, expected in test["original_result_sha256"].items():
        if file_sha256(args.reference / filename) != expected:
            raise ValueError("Original result changed")
    for directory, manifest_path in (
        (args.output / "development" / "predictions_mm", args.output / "development" / "prediction_manifest.json"),
        (args.reference / "predictions_mm", args.output / "test_source_prediction_manifest.json")):
        manifest = load_json(manifest_path)
        files = sorted(directory.glob("*_prediction_mm.npy"))
        if [p.name for p in files] != [item["file"] for item in manifest]:
            raise ValueError("Prediction image set mismatch")
        for item in manifest:
            if file_sha256(directory / item["file"]) != item["sha256"]:
                raise ValueError("Raw prediction modified")
    config = load_config(args.config)
    development = C3VDDataset(config["data_root"], "trans_t1_a")
    histogram = np.zeros(65536, dtype=np.int64)
    selected = []
    for index, pair in enumerate(development.pairs):
        raw, _, _ = development.load_depth(index)
        mask = evaluation_mask(raw)
        pred = np.load(args.output / "development" / "predictions_mm" / f"{pair.frame_id}_prediction_mm.npy", allow_pickle=False)
        if pred.dtype != np.float32 or pred.shape != raw.shape or not np.isfinite(pred).all() or (pred <= 0).any():
            raise ValueError("Invalid stored development array")
        selected.append(pred[mask])
        histogram += np.bincount(raw[mask], minlength=65536)
    pooled = np.concatenate(selected)
    del selected
    prediction_median = float(np.median(pooled, overwrite_input=True))
    del pooled
    count = int(histogram.sum())
    middle_codes = np.searchsorted(histogram.cumsum(), [(count - 1)//2, count//2], side="right")
    gt_median = float(np.mean(middle_codes) * 100 / 65535)
    np.testing.assert_allclose([gt_median, prediction_median, gt_median / prediction_median],
        [fit["gt_median_mm"], fit["prediction_median_mm"], fit["scale_factor"]], rtol=1e-14, atol=1e-14)
    if count != dev["original_metrics"]["valid_pixels"]:
        raise ValueError("Development median population mismatch")
    for filename, expected in (
        ("development_per_frame.csv", {"original": dev["original_metrics"], "global_scale": dev["calibrated_metrics"]}),
        ("test_per_frame.csv", test["results"])):
        with (args.output / filename).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for method, result in expected.items():
            relevant = [row for row in rows if row["method"] == method]
            n = sum(int(row["valid_pixels"]) for row in relevant)
            if n != result["valid_pixels"]:
                raise ValueError("Per-frame population mismatch")
            reference = [sum(float(r["abs_rel"]) * int(r["valid_pixels"]) for r in relevant) / n,
                np.sqrt(sum(float(r["rmse_mm"])**2 * int(r["valid_pixels"]) for r in relevant) / n),
                sum(float(r["delta_1_25"]) * int(r["valid_pixels"]) for r in relevant) / n]
            np.testing.assert_allclose(reference, [result[k] for k in ("abs_rel", "rmse_mm", "delta_1_25")],
                                       rtol=1e-12, atol=1e-12)
    calibration = GlobalScaleCalibration(fit["scale_factor"])
    for sequence, directory, csv_name in (
        ("trans_t1_a", args.output / "development" / "predictions_mm", "development_per_frame.csv"),
        ("trans_t2_b", args.reference / "predictions_mm", "test_per_frame.csv")):
        dataset = C3VDDataset(config["data_root"], sequence)
        with (args.output / csv_name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for index in (0, len(dataset)//2, len(dataset)-1):
            raw, gt, _ = dataset.load_depth(index)
            pair = dataset.pairs[index]
            original = np.load(directory / f"{pair.frame_id}_prediction_mm.npy", allow_pickle=False)
            mask = evaluation_mask(raw)
            prediction = calibration.apply(original)[mask].astype(np.float64)
            gt = gt[mask].astype(np.float64)
            reference = [np.mean(np.abs(prediction-gt)/gt), np.sqrt(np.mean((prediction-gt)**2)),
                         np.mean(np.maximum(prediction/gt, gt/prediction) < 1.25)]
            row = next(r for r in rows if r["method"] == "global_scale" and r["frame_id"] == pair.frame_id)
            np.testing.assert_allclose(reference, [float(row[k]) for k in ("abs_rel", "rmse_mm", "delta_1_25")],
                                       rtol=1e-12, atol=1e-12)
            if sequence == "trans_t1_a":
                with (args.output / "development_structure_per_frame.csv").open(newline="", encoding="utf-8") as handle:
                    correlation_rows = list(csv.DictReader(handle))
                correlation = float(np.corrcoef(gt, original[mask].astype(np.float64))[0, 1])
                recorded = next(r for r in correlation_rows if r["frame_id"] == pair.frame_id)
                np.testing.assert_allclose(correlation, float(recorded["pearson_r"]), rtol=1e-12, atol=1e-12)
    write_json(args.output / "verification.json", {"status": "passed", "development_predictions": 61,
        "cached_test_predictions_unchanged": 103, "inference_rerun": False,
        "checks": ["frozen scale fingerprints", "original result and raw prediction fingerprints unchanged",
                   "independent pooled median recomputation from development predictions",
                   "all per-frame to pooled metric reconstructions",
                   "direct formulas on first/middle/last frames of each split",
                   "independent numpy Pearson checks on fixed development samples"]})
    print("PASS: development medians, frozen scale, original artifacts, pooled metrics and fixed-sample correlations verified")


if __name__ == "__main__":
    main()
