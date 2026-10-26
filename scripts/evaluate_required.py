"""One frozen-protocol comparison: constant baseline and uncalibrated zero-shot model."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
import transformers

from src.baseline import ConstantMedianDepth
from src.c3vd_dataset import C3VDDataset
from src.depth_model import DepthAnythingV2Metric
from src.evaluation import evaluation_mask, file_sha256, read_frozen_artifacts
from src.metrics import DepthMetricAccumulator
from src.project import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/results"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    config = load_config(args.config)
    protocol, fit = read_frozen_artifacts(args.output)
    if config["model_id"] != protocol["model_id"] or config["model_revision"] != protocol["model_revision"]:
        raise ValueError("Model differs from frozen protocol")
    if (args.output / "summary.json").exists() or (args.output / "per_frame.csv").exists():
        raise FileExistsError("Evaluation outputs already exist; refusing to overwrite")
    dataset = C3VDDataset(config["data_root"], protocol["test_sequence"])
    if len(dataset) != protocol["expected_test_frames"]:
        raise ValueError(f"Expected 103 test frames, found {len(dataset)}")
    baseline = ConstantMedianDepth(fit["median_mm"])
    print("Loading cached model once...", flush=True)
    model = DepthAnythingV2Metric(config["model_id"], device=args.device, cache_dir=config["cache_dir"],
                                 revision=config["model_revision"], local_files_only=True)
    if model.revision != protocol["model_revision"]:
        raise ValueError("Loaded model revision differs from frozen revision")
    predictions = args.output / "predictions_mm"
    predictions.mkdir(exist_ok=True)
    accumulators = {name: DepthMetricAccumulator() for name in ("constant_median", "depth_anything_v2_small")}
    rows = []
    total_pixels, valid_pixels, elapsed_inference = 0, 0, 0.0
    started = time.perf_counter()
    write_json(args.output / "run_status.json", {"status": "running", "completed_frames": 0})
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            mask = evaluation_mask(sample.depth_raw)
            if sample.depth_mm.shape != (1080, 1350):
                raise ValueError(f"Unexpected native resolution: {sample.depth_mm.shape}")
            before = time.perf_counter()
            prediction = model.predict(sample.rgb)
            inference_seconds = time.perf_counter() - before
            elapsed_inference += inference_seconds
            # Save exactly the float32 output; no calibration, clipping or masking.
            np.save(predictions / f"{sample.pair.frame_id}_prediction_mm.npy", prediction, allow_pickle=False)
            if prediction.dtype != np.float32 or prediction.shape != sample.depth_mm.shape:
                raise ValueError(f"Invalid model output type/shape at {sample.pair.frame_id}")
            if not np.isfinite(prediction).all() or (prediction <= 0).any():
                raise ValueError(f"Invalid model prediction at {sample.pair.frame_id}; evaluation aborted")
            total_pixels += mask.size
            valid_pixels += int(mask.sum())
            for name, depth in (("constant_median", baseline.predict(mask.shape)),
                                ("depth_anything_v2_small", prediction)):
                metrics = accumulators[name].update(sample.depth_mm, depth, mask)
                rows.append({"method": name, "frame_id": sample.pair.frame_id,
                    "total_pixels": mask.size, **metrics,
                    "inference_seconds": inference_seconds if name == "depth_anything_v2_small" else ""})
            if (index + 1) % 10 == 0 or index == 0 or index == len(dataset) - 1:
                print(f"Evaluated {index+1}/{len(dataset)} test frames on {model.device}", flush=True)
        with (args.output / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        environment = {"python": platform.python_version(), "python_executable": sys.executable,
            "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "device": str(model.device),
            "gpu_name": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
            "transformers": transformers.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "precision": "float32, no autocast",
            "batch_size": 1}
        results = {name: accumulator.result() for name, accumulator in accumulators.items()}
        if len({v["valid_pixels"] for v in results.values()}) != 1:
            raise RuntimeError("Methods used different evaluation populations")
        summary = {"completed_at_singapore": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "status": "complete", "sequence": dataset.sequence, "frames": len(dataset),
            "frame_ids": [pair.frame_id for pair in dataset.pairs], "total_pixels": total_pixels,
            "valid_pixels": valid_pixels, "excluded_fraction": 1 - valid_pixels / total_pixels,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": file_sha256(args.output / "protocol.json"),
            "baseline_fit_sha256": file_sha256(args.output / "baseline_fit.json"),
            "baseline_median_mm": baseline.median_mm,
            "model_id": model.model_id, "model_revision": model.revision,
            "environment": environment, "results": results,
            "invalid_predictions_observed": 0, "predictions_modified": False,
            "test_gt_used_for_fitting_or_calibration": False,
            "inference_seconds_total": elapsed_inference,
            "evaluation_seconds_total": time.perf_counter() - started}
        write_json(args.output / "summary.json", summary)
        lines = ["# Required zero-shot comparison", "", "All valid pixels pooled across 103 test frames; no scaling.", "",
                 "| Method | AbsRel ↓ | RMSE (mm) ↓ | δ < 1.25 ↑ |",
                 "| --- | ---: | ---: | ---: |"]
        for key, label in (("constant_median", "Constant-Median-Depth"),
                           ("depth_anything_v2_small", "Depth Anything V2 Small")):
            value = results[key]
            lines.append(f"| {label} | {value['abs_rel']:.8f} | {value['rmse_mm']:.8f} | {value['delta_1_25']:.8f} |")
        (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(args.output / "run_status.json", {"status": "complete", "completed_frames": len(dataset)})
        print("\n".join(lines[4:]), flush=True)
    except Exception as error:
        write_json(args.output / "run_status.json", {"status": "failed", "error": str(error),
                   "completed_frames": len(rows) // 2,
                   "partial_results_must_not_be_reported_as_final": True})
        raise


if __name__ == "__main__":
    main()
