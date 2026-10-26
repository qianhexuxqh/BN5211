"""Official Transformers interface, raw metric output in millimeters."""

import os
from pathlib import Path
from typing import Iterable, Iterator

# Keep public model assets inside this authorized workspace.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"


class DepthAnythingV2Metric:
    def __init__(self, model_id=MODEL_ID, device="auto", cache_dir=None,
                 revision="main", local_files_only=False):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable in this Python environment")
        cache_dir = str(cache_dir or PROJECT_ROOT / ".cache" / "huggingface")
        options = dict(cache_dir=cache_dir, revision=revision,
                       local_files_only=local_files_only, trust_remote_code=False)
        self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False, **options)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, use_safetensors=True, **options)
        if self.model.config.depth_estimation_type != "metric":
            raise ValueError("Expected a metric-depth checkpoint")
        self.model.to(self.device).eval()
        self.model_id = model_id
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def predict(self, rgb: np.ndarray | Image.Image) -> np.ndarray:
        """One RGB image -> native HxW float32 Z-depth in mm; no calibration.

        Metric checkpoint output is meters; multiply by 1000 exactly once.
        Bicubic resizing follows the official model card. No visualization
        normalization, clipping to the GT range, or median scaling is applied.
        """
        if isinstance(rgb, np.ndarray):
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("RGB array must have shape HxWx3 and dtype uint8")
            rgb = Image.fromarray(rgb)
        if not isinstance(rgb, Image.Image):
            raise TypeError("Expected a PIL image or uint8 RGB array")
        rgb = rgb.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            depth_m = self.model(**inputs).predicted_depth
            resized_m = torch.nn.functional.interpolate(
                depth_m.unsqueeze(1), size=(rgb.height, rgb.width),
                mode="bicubic", align_corners=False,
            )[0, 0]
            prediction_mm = (resized_m * 1000.0).float().cpu().numpy()
        if prediction_mm.shape != (rgb.height, rgb.width) or not np.isfinite(prediction_mm).all():
            raise RuntimeError("Invalid prediction shape or nonfinite values")
        return prediction_mm

    def predict_many(self, images: Iterable[np.ndarray | Image.Image]) -> Iterator[np.ndarray]:
        """Future streaming interface; reuses the model, one image at a time."""
        for image in images:
            yield self.predict(image)
