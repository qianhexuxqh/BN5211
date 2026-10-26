"""Evaluate one frozen development scale on existing test predictions. No model import."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import numpy as np

from src.calibration import GlobalScaleCalibration
from src.c3vd_dataset import C3VDDataset
from src.evaluation import evaluation_mask, file_sha256, read_frozen_artifacts
from src.metrics import DepthMetricAccumulator
from src.project import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", type=Path, default=Path("outputs/results"))
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration"))
    args = parser.parse_args()
    if (args.output / "test_summary.json").exists() or (args.output / "test_per_frame.csv").exists():
        raise FileExistsError("Calibrated test evaluation already exists; refusing to repeat or overwrite")
    protocol, _ = read_frozen_artifacts(args.reference)
    fit_path = args.output / "scale_fit.json"
    frozen_hash = file_sha256(fit_path)
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    dev = json.loads((args.output / "development_summary.json").read_text(encoding="utf-8"))
    if fit["status"] != "frozen" or fit["fitted_sequence"] != "trans_t1_a" or not fit["frozen_before_calibrated_test_evaluation"]:
        raise ValueError("Development-only scale must be frozen first")
    if frozen_hash != dev["scale_fit_sha256"] or fit["protocol_sha256"] != file_sha256(args.reference / "protocol.json"):
        raise ValueError("Frozen scale or original evaluation protocol changed")
    config = load_config(args.config)
    source_summary_path = args.reference / "summary.json"
    source_per_frame_path = args.reference / "per_frame.csv"
    source_hashes = {path.name: file_sha256(path) for path in (source_summary_path, source_per_frame_path)}
    original_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if original_summary["model_revision"] != fit["model_revision"] or original_summary["status"] != "complete":
        raise ValueError("Original test result is incompatible")
    with source_per_frame_path.open(newline="", encoding="utf-8") as handle:
        original_rows = list(csv.DictReader(handle))
    dataset = C3VDDataset(config["data_root"], "trans_t2_b")
    frame_ids = [pair.frame_id for pair in dataset.pairs]
    if len(dataset) != 103 or frame_ids != original_summary["frame_ids"]:
        raise ValueError("Test image set differs from original experiment")
    calibration = GlobalScaleCalibration(fit["scale_factor"])
    calibrated_accumulator, original_check = DepthMetricAccumulator(), DepthMetricAccumulator()
    calibrated_rows, source_manifest = [], []
    for index in range(len(dataset)):
        raw, gt, _ = dataset.load_depth(index)
        pair = dataset.pairs[index]
        mask = evaluation_mask(raw)
        path = args.reference / "predictions_mm" / f"{pair.frame_id}_prediction_mm.npy"
        prediction = np.load(path, allow_pickle=False)
        if prediction.dtype != np.float32 or prediction.shape != gt.shape:
            raise ValueError(f"Bad saved original prediction: {path}")
        source_manifest.append({"frame_id": pair.frame_id, "file": path.name, "sha256": file_sha256(path)})
        original_check.update(gt, prediction, mask)
        calibrated = calibration.apply(prediction)
        metrics = calibrated_accumulator.update(gt, calibrated, mask)
        calibrated_rows.append({"method": "global_scale", "frame_id": pair.frame_id, **metrics})
        if index == 0 or (index + 1) % 20 == 0 or index == 102:
            print(f"Evaluated fixed calibration on cached test prediction {index+1}/103", flush=True)
    original_verified = original_check.result()
    expected_original = original_summary["results"]["depth_anything_v2_small"]
    for key in ("valid_pixels", "abs_rel", "rmse_mm", "delta_1_25"):
        np.testing.assert_allclose(original_verified[key], expected_original[key], rtol=1e-12, atol=1e-12)
    if frozen_hash != file_sha256(fit_path):
        raise RuntimeError("Scale changed during evaluation")
    for name, old_hash in source_hashes.items():
        if file_sha256(args.reference / name) != old_hash:
            raise RuntimeError("Original result was modified")
    results = {"constant_median": original_summary["results"]["constant_median"],
               "original_depth_anything_v2": expected_original,
               "global_scale": calibrated_accumulator.result()}
    if len({value["valid_pixels"] for value in results.values()}) != 1:
        raise RuntimeError("Methods used different evaluation populations")
    summary = {"status": "complete", "completed_at_singapore": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "sequence": "trans_t2_b", "frames": len(dataset), "frame_ids": frame_ids,
        "valid_pixels": calibrated_accumulator.valid_pixels, "protocol_id": protocol["protocol_id"],
        "protocol_sha256": fit["protocol_sha256"], "scale_fit_sha256": frozen_hash,
        "scale_factor": calibration.scale_factor, "model_id": fit["model_id"], "model_revision": fit["model_revision"],
        "original_results_directory": str(args.reference.resolve()), "original_result_sha256": source_hashes,
        "original_predictions_manifest_sha256": None,
        "model_inference_performed": False, "test_gt_used_for_fitting": False,
        "calibration_methods_evaluated": 1, "scale_refitted_after_test": False,
        "calibrated_arrays_stored": False,
        "calibrated_depth_reproduction": "GlobalScaleCalibration(scale_factor).apply(original_float32_mm)",
        "invalid_prediction_policy": "raise; no prediction-dependent filtering or clipping",
        "results": results}
    write_json(args.output / "test_source_prediction_manifest.json", source_manifest)
    summary["original_predictions_manifest_sha256"] = file_sha256(args.output / "test_source_prediction_manifest.json")
    write_json(args.output / "test_summary.json", summary)
    fields = ["method", "frame_id", "valid_pixels", "abs_rel", "rmse_mm", "delta_1_25"]
    copied = [{key: row[key] for key in fields} for row in original_rows]
    for row in copied:
        if row["method"] == "depth_anything_v2_small":
            row["method"] = "original_depth_anything_v2"
    with (args.output / "test_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(copied + calibrated_rows)
    lines = ["# Development-set global-scale calibration", "",
        "One factor from all valid development pixels; original test predictions reused without inference.", "",
        "| Method | AbsRel ↓ | RMSE (mm) ↓ | δ < 1.25 ↑ |", "| --- | ---: | ---: | ---: |"]
    for key, label in (("constant_median", "Constant-Median-Depth"),
                       ("original_depth_anything_v2", "Original Depth Anything V2"),
                       ("global_scale", "Depth Anything V2 + Global Scale Calibration")):
        value = results[key]
        lines.append(f"| {label} | {value['abs_rel']:.8f} | {value['rmse_mm']:.8f} | {value['delta_1_25']:.8f} |")
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(args.output / "run_status.json", {"status": "complete", "development_frames_inferred": 61,
        "test_predictions_reused": 103, "test_inference_performed": False})
    print("\n".join(lines[4:]), flush=True)


if __name__ == "__main__":
    main()
