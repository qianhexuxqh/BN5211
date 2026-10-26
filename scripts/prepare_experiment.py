"""Freeze the evaluation convention and fit the baseline using development GT only."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

from src.baseline import ConstantMedianDepth
from src.c3vd_dataset import C3VDDataset
from src.evaluation import PROTOCOL_ID, evaluation_mask, interior_depth_mask, file_sha256
from src.project import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/results"))
    args = parser.parse_args()
    config = load_config(args.config)
    if config["development_sequence"] != "trans_t1_a" or config["test_sequence"] != "trans_t2_b":
        raise ValueError("Course split must remain fixed")
    if (args.output / "protocol.json").exists():
        raise FileExistsError("Frozen protocol already exists; use another output directory for a new protocol")
    dataset = C3VDDataset(config["data_root"], "trans_t1_a", mask_policy=interior_depth_mask)
    if len(dataset) != 61:
        raise ValueError(f"Expected 61 development frames, found {len(dataset)}")
    rows = []
    total, zeros, saturated = 0, 0, 0
    for index, pair in enumerate(dataset.pairs):
        raw, _, mask = dataset.load_depth(index)
        if not np.array_equal(mask, evaluation_mask(raw)):
            raise RuntimeError("Decoded and raw masking policies disagree")
        z, s, n = int(np.count_nonzero(raw == 0)), int(np.count_nonzero(raw == 65535)), raw.size
        rows.append({"frame_id": pair.frame_id, "total_pixels": n, "zero_pixels": z,
                     "saturated_pixels": s, "valid_pixels": int(mask.sum()),
                     "excluded_fraction": (z + s) / n})
        total += n
        zeros += z
        saturated += s
    # Reuse the existing exact histogram implementation. No test sequence is opened.
    baseline = ConstantMedianDepth.fit_development(dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "frozen_at_singapore": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "frozen_before_test_evaluation": True,
        "development_sequence": "trans_t1_a", "test_sequence": "trans_t2_b",
        "expected_development_frames": 61, "expected_test_frames": 103,
        "mask_raw": "0 < depth_raw < 65535",
        "mask_authority": "project convention, not an official C3VD mandated evaluation mask",
        "source": "https://durrlab.github.io/C3VD/",
        "rationale": "Zero cannot support ratio metrics; 65535 is the 100 mm clamping boundary, "
                     "so evaluate positive interior non-saturated measurements. No crop, reflection mask "
                     "or prediction-dependent filtering.",
        "depth_conversion": "uint16_code * 100 / 65535 = mm; model meters * 1000 = mm",
        "aggregation": "pool every valid pixel across all 103 test frames",
        "abs_rel": "sum(abs(pred-gt)/gt) / N",
        "rmse_mm": "sqrt(sum((pred-gt)^2) / N)",
        "delta_1_25": "count(max(pred/gt,gt/pred) < 1.25) / N",
        "arithmetic": "float64 metric accumulation; float32 stored predictions",
        "invalid_prediction_policy": "fail explicitly on any nonfinite or nonpositive predicted value; "
                                     "do not alter mask, clip predictions or compute partial final scores",
        "scale_calibration": False, "training_or_finetuning": False,
        "development_mask_statistics": {"frames": len(dataset), "total_pixels": total,
            "zero_pixels": zeros, "saturated_pixels": saturated,
            "valid_pixels": total - zeros - saturated,
            "excluded_fraction": (zeros + saturated) / total,
            "zero_fraction": zeros / total, "saturated_fraction": saturated / total},
        "model_id": config["model_id"], "model_revision": config["model_revision"],
    }
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "baseline_fit.json", {
        "fitted_sequence": "trans_t1_a", "frames": len(dataset),
        "frame_ids": [pair.frame_id for pair in dataset.pairs],
        "median_mm": baseline.median_mm, "prediction_dtype": "float32",
        "effective_prediction_mm": float(np.float32(baseline.median_mm)),
        "method": "exact global median of pooled masked uint16 pixels, then decode to mm",
        "valid_pixels": total - zeros - saturated,
        "protocol_id": PROTOCOL_ID, "protocol_sha256": file_sha256(args.output / "protocol.json"),
        "test_gt_used_for_fitting": False,
    })
    with (args.output / "development_mask_per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Frozen protocol: {PROTOCOL_ID}; development exclusion={(zeros+saturated)/total:.6%}")
    print(f"Baseline global median: {baseline.median_mm:.12f} mm; {len(dataset)} development frames only")


if __name__ == "__main__":
    main()
