"""Fit the prespecified log-affine model on cached development predictions only."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import platform
import sys
import numpy as np

from src.calibration import GlobalScaleCalibration
from src.contrast_calibration import StreamingLogLeastSquares, transformed_quantiles_inplace
from src.c3vd_dataset import C3VDDataset
from src.evaluation import evaluation_mask, file_sha256, read_frozen_artifacts
from src.metrics import DepthMetricAccumulator
from src.project import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", type=Path, default=Path("outputs/results"))
    parser.add_argument("--previous", type=Path, default=Path("outputs/calibration"))
    parser.add_argument("--output", type=Path, default=Path("outputs/contrast_calibration"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Contrast-calibration output already exists; refusing to refit or overwrite")
    protocol, _ = read_frozen_artifacts(args.reference)
    config = load_config(args.config)
    prior_dev_path, scale_path = args.previous / "development_summary.json", args.previous / "scale_fit.json"
    prior_dev = json.loads(prior_dev_path.read_text(encoding="utf-8"))
    scale_fit = json.loads(scale_path.read_text(encoding="utf-8"))
    if prior_dev["scale_fit_sha256"] != file_sha256(scale_path):
        raise ValueError("Previous development-scale fit changed")
    if scale_fit["protocol_sha256"] != file_sha256(args.reference / "protocol.json"):
        raise ValueError("Frozen protocol differs from the previous calibration")
    dataset = C3VDDataset(config["data_root"], "trans_t1_a")
    if len(dataset) != 61:
        raise ValueError("Expected 61 development frames")
    args.output.mkdir(parents=True)
    # Temporary local mmap supports exact quantiles without collecting all
    # development pixels in a Python RAM array. Original files are read only.
    expected_pixels = prior_dev["original_metrics"]["valid_pixels"]
    pool_path = args.output / "temporary_valid_prediction_pool.npy"
    pool = np.lib.format.open_memmap(pool_path, mode="w+", dtype=np.float32, shape=(expected_pixels,))
    statistics = StreamingLogLeastSquares()
    prediction_dir = args.previous / "development" / "predictions_mm"
    offset = 0
    for index, pair in enumerate(dataset.pairs):
        raw, gt, _ = dataset.load_depth(index)
        mask = evaluation_mask(raw)
        original = np.load(prediction_dir / f"{pair.frame_id}_prediction_mm.npy", allow_pickle=False)
        if original.shape != gt.shape or original.dtype != np.float32 or not np.isfinite(original).all() or (original <= 0).any():
            raise ValueError(f"Invalid cached development prediction: {pair.frame_id}")
        statistics.update(gt, original, mask)
        selected = original[mask]
        pool[offset:offset + selected.size] = selected
        offset += selected.size
        if index == 0 or (index + 1) % 20 == 0 or index == 60:
            print(f"Accumulated development log statistics {index+1}/61", flush=True)
    if offset != expected_pixels or statistics.n != expected_pixels:
        raise ValueError("Development pixel population differs from the frozen protocol")
    calibration = statistics.fit_development(sequence=dataset.sequence)
    fit = {"status": "frozen", "frozen_at_singapore": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "a": calibration.a, "b": calibration.b,
        "formula": "exp(a + b * natural_log(original_depth_mm))",
        "objective": "minimize sum((natural_log(GT_mm) - a - b * natural_log(prediction_mm))**2)",
        "method": "streaming float64 centered second moments merged per frame; b=Sxy/Sxx, a=mean_y-b*mean_x",
        "fitted_sequence": "trans_t1_a", "development_frames": len(dataset),
        "statistics": statistics.diagnostics(calibration),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": file_sha256(args.reference / "protocol.json"),
        "model_id": scale_fit["model_id"], "model_revision": scale_fit["model_revision"],
        "development_prediction_directory": str(prediction_dir.resolve()),
        "previous_scale_fit_sha256": file_sha256(scale_path),
        "frozen_before_test_evaluation": True, "test_data_used_for_fitting": False,
        "parameter_search": False, "other_new_methods_tried": 0,
        "output_dtype": "float64", "clipping_or_normalization": False,
        "invalid_value_policy": "raise on nonfinite/nonpositive input or overflow/underflow; no filtering"}
    # Freeze once, before the independent test script can access test arrays.
    write_json(args.output / "log_affine_fit.json", fit)
    scale = GlobalScaleCalibration(scale_fit["scale_factor"])
    quantiles = {
        "ground_truth_mm": {key: prior_dev["ground_truth_distribution"]["quantiles_mm"][key]
                            for key in ("p05", "median", "p95")},
        "global_scale_mm": transformed_quantiles_inplace(pool, scale.apply),
        "log_affine_mm": transformed_quantiles_inplace(pool, calibration.apply, decreasing=calibration.b < 0),
        "definition": "exact all-valid-pixel linear empirical quantiles; temporary mmap partition; nonlinear transform before interpolation"}
    # Close and remove only the temporary pool created by this run.
    pool.flush()
    del pool
    pool_path.unlink()
    accumulators = {method: DepthMetricAccumulator() for method in ("original", "global_scale", "log_affine")}
    rows = []
    extrema = {"min_mm": float("inf"), "max_mm": 0., "above_100_mm_valid_pixels": 0}
    log_sse = 0.0
    for index, pair in enumerate(dataset.pairs):
        raw, gt, _ = dataset.load_depth(index)
        mask = evaluation_mask(raw)
        original = np.load(prediction_dir / f"{pair.frame_id}_prediction_mm.npy", allow_pickle=False)
        corrected = calibration.apply(original)
        log_residual = np.log(gt[mask].astype(np.float64)) - np.log(corrected[mask])
        log_sse += float(np.dot(log_residual, log_residual))
        extrema["min_mm"] = min(extrema["min_mm"], float(corrected[mask].min()))
        extrema["max_mm"] = max(extrema["max_mm"], float(corrected[mask].max()))
        extrema["above_100_mm_valid_pixels"] += int(np.count_nonzero(corrected[mask] > 100))
        for method, prediction in (("original", original), ("global_scale", scale.apply(original)), ("log_affine", corrected)):
            rows.append({"method": method, "frame_id": pair.frame_id,
                         **accumulators[method].update(gt, prediction, mask)})
    results = {method: accumulator.result() for method, accumulator in accumulators.items()}
    for method, expected in (("original", prior_dev["original_metrics"]), ("global_scale", prior_dev["calibrated_metrics"])):
        for key in ("valid_pixels", "abs_rel", "rmse_mm", "delta_1_25"):
            np.testing.assert_allclose(results[method][key], expected[key], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(log_sse, fit["statistics"]["fitted_log_sse"], rtol=1e-12)
    summary = {"status": "complete", "sequence": "trans_t1_a", "frames": len(dataset),
        "valid_pixels": expected_pixels, "frame_ids": [p.frame_id for p in dataset.pairs],
        "results": results, "depth_quantiles": quantiles, "log_affine_extrema": extrema,
        "direct_log_sse": log_sse, "protocol_id": protocol["protocol_id"],
        "log_affine_fit_sha256": file_sha256(args.output / "log_affine_fit.json"),
        "model_inference_performed": False,
        "environment": {"python_executable": sys.executable, "python": platform.python_version(),
                        "numpy": np.__version__, "fitting_and_application_dtype": "float64", "compute_device": "CPU"}}
    write_json(args.output / "development_summary.json", summary)
    with (args.output / "development_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "run_status.json", {"status": "development_complete_parameters_frozen",
        "development_cached_frames_used": 61, "inference_performed": False})
    print(f"Frozen a={calibration.a:.17g}, b={calibration.b:.17g}")
    print(f"Development results: {results}")
    print(f"Depth quantiles: {quantiles}")


if __name__ == "__main__":
    main()
