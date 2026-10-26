"""Frozen project evaluation convention, independent of model predictions."""

import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL_ID = "c3vd-interior-v1"


def interior_depth_mask(depth_mm: np.ndarray) -> np.ndarray:
    """Dataset mask callback, equivalent to 0 < uint16_code < 65535."""
    return np.isfinite(depth_mm) & (depth_mm > 0) & (depth_mm < 100)


def evaluation_mask(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError("Evaluation mask requires 2D uint16 C3VD depth")
    return (raw > 0) & (raw < 65535)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frozen_artifacts(directory: Path):
    protocol_path, baseline_path = directory / "protocol.json", directory / "baseline_fit.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != PROTOCOL_ID or protocol["mask_raw"] != "0 < depth_raw < 65535":
        raise ValueError("Unsupported evaluation protocol")
    if baseline["fitted_sequence"] != "trans_t1_a":
        raise ValueError("Baseline was not fitted on the development split")
    if baseline["protocol_sha256"] != file_sha256(protocol_path):
        raise ValueError("Protocol changed after baseline fitting")
    if not protocol["frozen_before_test_evaluation"]:
        raise ValueError("Protocol must be frozen before test evaluation")
    return protocol, baseline
