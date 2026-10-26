"""One development image, one forward pass; no accuracy metrics."""

import argparse
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch
import transformers

from src.c3vd_dataset import C3VDDataset
from src.depth_model import DepthAnythingV2Metric
from src.project import load_config, save_depth_preview, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/model_smoke"))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = C3VDDataset(config["data_root"], config["development_sequence"])
    sample = dataset[0]
    started = time.perf_counter()
    print("Loading metric depth checkpoint...", flush=True)
    model = DepthAnythingV2Metric(
        config["model_id"], device=args.device or config["device"],
        cache_dir=config["cache_dir"], revision=config["model_revision"],
        local_files_only=args.local_files_only,
    )
    loading_seconds = time.perf_counter() - started
    print(f"Model loaded on {model.device}; running one development image...", flush=True)
    started = time.perf_counter()
    prediction_mm = model.predict(sample.rgb)
    inference_seconds = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "prediction_mm.npy", prediction_mm, allow_pickle=False)
    preview = save_depth_preview(prediction_mm, args.output / "prediction_preview.png")
    report = {"purpose": "single-image smoke test, not a formal experiment",
              "sequence": dataset.sequence, "frame_id": sample.pair.frame_id,
              "rgb_path": str(sample.pair.rgb_path), "model_id": model.model_id,
              "model_revision": model.revision, "device": str(model.device),
              "python": platform.python_version(), "python_executable": sys.executable,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "cuda_available": torch.cuda.is_available(), "model_loaded": True,
              "inference_succeeded": True, "shape": list(prediction_mm.shape),
              "dtype": str(prediction_mm.dtype), "all_finite": bool(np.isfinite(prediction_mm).all()),
              "minimum_mm": float(prediction_mm.min()), "maximum_mm": float(prediction_mm.max()),
              "unit_conversion": "raw model meters * 1000 = millimeters; no scale calibration",
              "loading_seconds": loading_seconds, "inference_seconds": inference_seconds,
              "visualization_only": preview, "metrics_computed": False}
    write_json(args.output / "smoke_test.json", report)
    print(f"PASS: {prediction_mm.shape}, {prediction_mm.dtype}, finite mm depth; saved to {args.output}")


if __name__ == "__main__":
    main()
