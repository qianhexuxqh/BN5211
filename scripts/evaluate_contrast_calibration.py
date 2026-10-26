"""Evaluate one frozen log-affine calibration on cached test predictions only."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import numpy as np

from src.contrast_calibration import LogAffineCalibration
from src.c3vd_dataset import C3VDDataset
from src.evaluation import evaluation_mask, file_sha256
from src.metrics import DepthMetricAccumulator
from src.project import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", type=Path, default=Path("outputs/results"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/calibration"))
    parser.add_argument("--output", type=Path, default=Path("outputs/contrast_calibration"))
    args = parser.parse_args()
    if (args.output / "test_summary.json").exists() or (args.output / "test_per_frame.csv").exists():
        raise FileExistsError("Test comparison already exists; refusing to repeat or overwrite")
    fit_path = args.output / "log_affine_fit.json"
    fit_hash = file_sha256(fit_path)
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    development = json.loads((args.output / "development_summary.json").read_text(encoding="utf-8"))
    if fit["status"] != "frozen" or fit["fitted_sequence"] != "trans_t1_a" or not fit["frozen_before_test_evaluation"]:
        raise ValueError("Parameters must be frozen from development only")
    if development["log_affine_fit_sha256"] != fit_hash:
        raise ValueError("Parameters changed after development evaluation")
    if fit["protocol_sha256"] != file_sha256(args.reference / "protocol.json"):
        raise ValueError("Evaluation protocol changed")
    if fit["previous_scale_fit_sha256"] != file_sha256(args.previous / "scale_fit.json"):
        raise ValueError("Previous scale calibration changed")
    sources = [args.reference / "summary.json", args.reference / "per_frame.csv",
               args.previous / "test_summary.json", args.previous / "test_per_frame.csv"]
    source_hashes = {str(path): file_sha256(path) for path in sources}
    previous = json.loads((args.previous / "test_summary.json").read_text(encoding="utf-8"))
    if previous["model_revision"] != fit["model_revision"] or previous["status"] != "complete":
        raise ValueError("Prior model result differs from this calibration")
    dataset = C3VDDataset(load_config(args.config)["data_root"], "trans_t2_b")
    if len(dataset) != 103 or [p.frame_id for p in dataset.pairs] != previous["frame_ids"]:
        raise ValueError("Test image set differs from existing experiments")
    calibration = LogAffineCalibration(fit["a"], fit["b"])
    accumulator = DepthMetricAccumulator()
    rows = []
    extrema = {"min_mm": float("inf"), "max_mm": 0., "above_100_mm_valid_pixels": 0}
    for index, pair in enumerate(dataset.pairs):
        raw, gt, _ = dataset.load_depth(index)
        mask = evaluation_mask(raw)
        original = np.load(args.reference / "predictions_mm" / f"{pair.frame_id}_prediction_mm.npy", allow_pickle=False)
        if original.shape != gt.shape or original.dtype != np.float32:
            raise ValueError(f"Invalid cached test output: {pair.frame_id}")
        corrected = calibration.apply(original)
        metrics = accumulator.update(gt, corrected, mask)
        rows.append({"method": "log_affine", "frame_id": pair.frame_id, **metrics})
        extrema["min_mm"] = min(extrema["min_mm"], float(corrected[mask].min()))
        extrema["max_mm"] = max(extrema["max_mm"], float(corrected[mask].max()))
        extrema["above_100_mm_valid_pixels"] += int(np.count_nonzero(corrected[mask] > 100))
        if index == 0 or (index + 1) % 20 == 0 or index == 102:
            print(f"Evaluated fixed log-affine calibration on cached test frame {index+1}/103", flush=True)
    if accumulator.valid_pixels != previous["valid_pixels"]:
        raise ValueError("Methods used different GT-valid pixel populations")
    results = {**previous["results"], "log_affine": accumulator.result()}
    global_result, new_result = results["global_scale"], results["log_affine"]
    changes = {"abs_rel_difference": new_result["abs_rel"] - global_result["abs_rel"],
        "abs_rel_relative_change": new_result["abs_rel"] / global_result["abs_rel"] - 1,
        "rmse_mm_difference": new_result["rmse_mm"] - global_result["rmse_mm"],
        "rmse_relative_change": new_result["rmse_mm"] / global_result["rmse_mm"] - 1,
        "delta_1_25_difference": new_result["delta_1_25"] - global_result["delta_1_25"],
        "delta_percentage_point_change": (new_result["delta_1_25"] - global_result["delta_1_25"]) * 100}
    if file_sha256(fit_path) != fit_hash:
        raise RuntimeError("Parameters changed during test evaluation")
    for path in sources:
        if file_sha256(path) != source_hashes[str(path)]:
            raise RuntimeError("A previous result file was modified")
    summary = {"status": "complete", "completed_at_singapore": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "sequence": "trans_t2_b", "frames": len(dataset), "frame_ids": [p.frame_id for p in dataset.pairs],
        "valid_pixels": accumulator.valid_pixels, "protocol_id": fit["protocol_id"],
        "protocol_sha256": fit["protocol_sha256"], "log_affine_fit_sha256": fit_hash,
        "a": fit["a"], "b": fit["b"], "model_id": fit["model_id"], "model_revision": fit["model_revision"],
        "results": results, "changes_vs_global_scale": changes, "log_affine_extrema": extrema,
        "prior_result_fingerprints": source_hashes,
        "model_inference_performed": False, "test_gt_used_for_fitting": False,
        "new_calibration_methods_evaluated": 1, "parameters_refitted_after_test": False,
        "predictions_clipped_or_normalized": False, "calibrated_arrays_stored": False,
        "application_dtype": "float64", "compute_device": "CPU; no model or CUDA usage"}
    write_json(args.output / "test_summary.json", summary)
    fields = ["method", "frame_id", "valid_pixels", "abs_rel", "rmse_mm", "delta_1_25"]
    with (args.previous / "test_per_frame.csv").open(newline="", encoding="utf-8") as handle:
        previous_rows = [{key: row[key] for key in fields} for row in csv.DictReader(handle)]
    with (args.output / "test_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(previous_rows + rows)
    lines = ["# Global log-affine calibration comparison", "",
             "Frozen development-only OLS in natural-log mm depth; no inference or clipping.", "",
             "| Method | AbsRel ↓ | RMSE (mm) ↓ | δ < 1.25 ↑ |", "| --- | ---: | ---: | ---: |"]
    for key, label in (("constant_median", "Constant-Median-Depth"),
                       ("original_depth_anything_v2", "Original Depth Anything V2"),
                       ("global_scale", "Global Scale Calibration"), ("log_affine", "Log-Affine Calibration")):
        result = results[key]
        lines.append(f"| {label} | {result['abs_rel']:.8f} | {result['rmse_mm']:.8f} | {result['delta_1_25']:.8f} |")
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(args.output / "run_status.json", {"status": "complete", "development_cached_frames": 61,
        "test_cached_frames": 103, "model_inference_performed": False})
    print("\n".join(lines[4:]), flush=True)
    print(f"Changes vs global scale: {changes}", flush=True)


if __name__ == "__main__":
    main()
