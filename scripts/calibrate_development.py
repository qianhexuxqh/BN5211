"""Infer 61 development frames, diagnose structure, fit and freeze one scale."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
import platform
import sys
import numpy as np
import torch
import transformers

from src.calibration import (GlobalScaleCalibration, histogram_depth_distribution,
    pooled_prediction_distribution, pearson_structure, summarize_structure)
from src.c3vd_dataset import C3VDDataset
from src.depth_model import DepthAnythingV2Metric
from src.evaluation import evaluation_mask, file_sha256, read_frozen_artifacts
from src.metrics import DepthMetricAccumulator
from src.project import load_config, write_json


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", type=Path, default=Path("outputs/results"))
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration"))
    args = parser.parse_args()
    config = load_config(args.config)
    protocol, _ = read_frozen_artifacts(args.reference)
    if config["model_id"] != protocol["model_id"] or config["model_revision"] != protocol["model_revision"]:
        raise ValueError("Model differs from frozen experiment")
    if args.output.exists():
        raise FileExistsError("Calibration output already exists; refusing to refit or overwrite")
    dataset = C3VDDataset(config["data_root"], "trans_t1_a")
    if len(dataset) != 61:
        raise ValueError("Expected 61 development frames")
    prediction_dir = args.output / "development" / "predictions_mm"
    prediction_dir.mkdir(parents=True)
    print("Loading existing checkpoint on CUDA once; development split only...", flush=True)
    model = DepthAnythingV2Metric(config["model_id"], device="cuda", cache_dir=config["cache_dir"],
        revision=config["model_revision"], local_files_only=True)
    original = DepthMetricAccumulator()
    original_rows, correlations, selected_predictions = [], [], []
    histogram = np.zeros(65536, dtype=np.int64)
    manifest = []
    write_json(args.output / "run_status.json", {"status": "development_inference", "completed_frames": 0})
    for index in range(len(dataset)):
        sample = dataset[index]
        mask = evaluation_mask(sample.depth_raw)
        prediction = model.predict(sample.rgb)
        if prediction.dtype != np.float32 or prediction.shape != sample.depth_mm.shape or (prediction <= 0).any():
            raise ValueError(f"Invalid development prediction at frame {sample.pair.frame_id}")
        path = prediction_dir / f"{sample.pair.frame_id}_prediction_mm.npy"
        np.save(path, prediction, allow_pickle=False)
        manifest.append({"frame_id": sample.pair.frame_id, "file": path.name, "sha256": file_sha256(path)})
        original_rows.append({"method": "original", "frame_id": sample.pair.frame_id,
                              **original.update(sample.depth_mm, prediction, mask)})
        correlations.append({"frame_id": sample.pair.frame_id,
                             **pearson_structure(sample.depth_mm, prediction, mask)})
        histogram += np.bincount(sample.depth_raw[mask], minlength=65536)
        selected_predictions.append(prediction[mask])
        if index == 0 or (index + 1) % 10 == 0 or index == 60:
            print(f"Saved development prediction {index+1}/61", flush=True)
    gt_distribution = histogram_depth_distribution(histogram)
    pred_distribution = pooled_prediction_distribution(selected_predictions)
    del selected_predictions
    if gt_distribution["valid_pixels"] != pred_distribution["valid_pixels"]:
        raise RuntimeError("Different median populations")
    gt_median = gt_distribution["quantiles_mm"]["median"]
    pred_median = pred_distribution["quantiles_mm"]["median"]
    calibration = GlobalScaleCalibration.fit_development(gt_median, pred_median, sequence=dataset.sequence)
    write_json(args.output / "development" / "prediction_manifest.json", manifest)
    fit = {"status": "frozen", "frozen_at_singapore": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "fitted_sequence": "trans_t1_a", "development_frames": len(dataset),
        "valid_pixels": gt_distribution["valid_pixels"],
        "formula": "median(GT_valid_development) / median(Prediction_valid_development)",
        "median_population": "all valid pixels pooled across all 61 development frames, not frame medians",
        "gt_median_mm": gt_median, "prediction_median_mm": pred_median,
        "scale_factor": calibration.scale_factor,
        "prediction_to_gt_median_ratio": pred_median / gt_median,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_sha256(args.reference / "protocol.json"),
        "development_prediction_manifest_sha256": file_sha256(args.output / "development" / "prediction_manifest.json"),
        "model_id": model.model_id, "model_revision": model.revision,
        "test_gt_used_for_fitting": False, "test_predictions_used_for_fitting": False,
        "frozen_before_calibrated_test_evaluation": True,
        "application": "same unique scale for every pixel of every image, float64 multiply then float32 storage; no clipping",
        "other_calibration_formulas_tried": 0}
    # Freeze the coefficient before the test script can access test arrays.
    write_json(args.output / "scale_fit.json", fit)
    calibrated = DepthMetricAccumulator()
    calibrated_rows = []
    for index, pair in enumerate(dataset.pairs):
        raw, gt, _ = dataset.load_depth(index)
        prediction = np.load(prediction_dir / f"{pair.frame_id}_prediction_mm.npy", allow_pickle=False)
        metrics = calibrated.update(gt, calibration.apply(prediction), evaluation_mask(raw))
        calibrated_rows.append({"method": "global_scale", "frame_id": pair.frame_id, **metrics})
    environment = {"python": platform.python_version(), "python_executable": sys.executable,
        "torch": torch.__version__, "cuda_build": torch.version.cuda, "device": str(model.device),
        "gpu": torch.cuda.get_device_name(model.device), "transformers": transformers.__version__,
        "numpy": np.__version__, "batch_size": 1, "precision": "float32; no autocast"}
    summary = {"status": "complete", "sequence": "trans_t1_a", "frames": len(dataset),
        "protocol_id": protocol["protocol_id"], "scale_fit_sha256": file_sha256(args.output / "scale_fit.json"),
        "ground_truth_distribution": gt_distribution, "prediction_distribution": pred_distribution,
        "original_metrics": original.result(), "calibrated_metrics": calibrated.result(),
        "structure": summarize_structure(correlations), "environment": environment}
    write_json(args.output / "development_summary.json", summary)
    write_csv(args.output / "development_per_frame.csv", original_rows + calibrated_rows)
    write_csv(args.output / "development_structure_per_frame.csv", correlations)
    write_json(args.output / "run_status.json", {"status": "development_complete_scale_frozen", "development_frames": 61})
    print(f"GT median={gt_median:.12f} mm; prediction median={pred_median:.12f} mm; scale={calibration.scale_factor:.15g}")
    print(f"Original metrics: {original.result()}")
    print(f"Calibrated metrics: {calibrated.result()}")
    print(f"Structure: {summary['structure']}")


if __name__ == "__main__":
    main()
