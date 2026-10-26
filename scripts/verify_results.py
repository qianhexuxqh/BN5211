"""Verify saved artifacts and pooled scores; no inference or parameter fitting."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from src.c3vd_dataset import C3VDDataset
from src.evaluation import evaluation_mask, file_sha256, read_frozen_artifacts
from src.project import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--results", type=Path, default=Path("outputs/results"))
    args = parser.parse_args()
    protocol, fit = read_frozen_artifacts(args.results)
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["frames"] != 103:
        raise ValueError("Incomplete final evaluation")
    if summary["protocol_sha256"] != file_sha256(args.results / "protocol.json"):
        raise ValueError("Summary protocol fingerprint mismatch")
    if summary["baseline_fit_sha256"] != file_sha256(args.results / "baseline_fit.json"):
        raise ValueError("Baseline fingerprint mismatch")
    with (args.results / "per_frame.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 206:
        raise ValueError("Expected 2 methods x 103 per-frame rows")
    for method, result in summary["results"].items():
        selected = [row for row in rows if row["method"] == method]
        if [row["frame_id"] for row in selected] != summary["frame_ids"]:
            raise ValueError("Per-frame image sets differ")
        weights = np.array([int(row["valid_pixels"]) for row in selected], dtype=np.int64)
        n = int(weights.sum())
        pooled_absrel = sum(float(row["abs_rel"]) * int(row["valid_pixels"]) for row in selected) / n
        pooled_rmse = np.sqrt(sum(float(row["rmse_mm"]) ** 2 * int(row["valid_pixels"])
                                  for row in selected) / n)
        pooled_delta = sum(float(row["delta_1_25"]) * int(row["valid_pixels"]) for row in selected) / n
        np.testing.assert_allclose([pooled_absrel, pooled_rmse, pooled_delta],
            [result["abs_rel"], result["rmse_mm"], result["delta_1_25"]], rtol=1e-12, atol=1e-12)
        if n != result["valid_pixels"] or n != summary["valid_pixels"]:
            raise ValueError("Pixel population mismatch")
    files = sorted((args.results / "predictions_mm").glob("*_prediction_mm.npy"))
    expected = [f"{frame_id}_prediction_mm.npy" for frame_id in summary["frame_ids"]]
    if [path.name for path in files] != expected:
        raise ValueError("Missing or unexpected prediction files")
    for path in files:
        array = np.load(path, allow_pickle=False)
        if array.shape != (1080, 1350) or array.dtype != np.float32:
            raise ValueError(f"Invalid stored output: {path}")
        if not np.isfinite(array).all() or (array <= 0).any():
            raise ValueError(f"Nonfinite/nonpositive stored prediction: {path}")
    # Direct formulas on fixed first/middle/last files, separate from accumulator.
    config = load_config(args.config)
    dataset = C3VDDataset(config["data_root"], protocol["test_sequence"])
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[index]
        mask = evaluation_mask(sample.depth_raw)
        gt = sample.depth_mm[mask].astype(np.float64)
        stored = np.load(args.results / "predictions_mm" / f"{sample.pair.frame_id}_prediction_mm.npy",
                         allow_pickle=False)
        for method, prediction in (("constant_median", np.full(gt.shape, np.float32(fit["median_mm"]), dtype=np.float64)),
                                   ("depth_anything_v2_small", stored[mask].astype(np.float64))):
            reference = [np.mean(np.abs(prediction-gt)/gt),
                         np.sqrt(np.mean((prediction-gt)**2)),
                         np.mean(np.maximum(prediction/gt, gt/prediction) < 1.25)]
            row = next(row for row in rows if row["method"] == method and row["frame_id"] == sample.pair.frame_id)
            np.testing.assert_allclose(reference, [float(row[k]) for k in ("abs_rel", "rmse_mm", "delta_1_25")],
                                       rtol=1e-12, atol=1e-12)
    write_json(args.results / "verification.json", {
        "status": "passed", "predictions_checked": len(files), "per_frame_rows": len(rows),
        "checks": ["protocol and baseline fingerprints", "identical image sets and pixel counts",
                   "all saved prediction shapes/dtypes/finite-positive values",
                   "pooled scores reconstructed from weighted per-frame metrics",
                   "direct formulas on first/middle/last stored predictions for both methods"],
        "inference_rerun": False,
    })
    print("PASS: 103 saved predictions, 206 per-frame rows, frozen protocol and pooled metrics verified")


if __name__ == "__main__":
    main()
