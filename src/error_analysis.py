"""Descriptive test error analysis only; no fitting or prediction modification."""

import numpy as np

DEPTH_EDGES_MM = np.array([0., 10., 20., 40., 60., 100.])
DEPTH_LABELS = ("0–10", "10–20", "20–40", "40–60", "60–100")


def depth_bin_indices(valid_gt_mm):
    gt = np.asarray(valid_gt_mm)
    if not np.isfinite(gt).all() or (gt <= 0).any() or (gt >= 100).any():
        raise ValueError("Depth analysis expects fixed-mask GT strictly inside (0,100) mm")
    # [0,10), [10,20), [20,40), [40,60), [60,100); raw endpoints excluded upstream.
    return np.searchsorted(DEPTH_EDGES_MM[1:-1], gt, side="right")


def delta_transitions(gt, global_scale, log_affine):
    gt, scale, log = map(lambda a: np.asarray(a, dtype=np.float64), (gt, global_scale, log_affine))
    if gt.shape != scale.shape or gt.shape != log.shape:
        raise ValueError("Delta transition inputs must have identical shapes")
    if not all(np.isfinite(a).all() and (a > 0).all() for a in (gt, scale, log)):
        raise ValueError("Delta transition inputs must be finite and positive")
    scale_pass = np.maximum(scale / gt, gt / scale) < 1.25
    log_pass = np.maximum(log / gt, gt / log) < 1.25
    return {"both_pass": scale_pass & log_pass, "global_only_pass": scale_pass & ~log_pass,
            "log_only_pass": ~scale_pass & log_pass, "both_fail": ~scale_pass & ~log_pass}


def tail_statistics(gt, prediction, thresholds=(50., 100.)):
    gt, prediction = np.asarray(gt, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    error = prediction - gt
    squared = error * error
    return {threshold: {"pixels": int(np.count_nonzero(np.abs(error) > threshold)),
                        "squared_error_sum_mm2": float(np.sum(squared[np.abs(error) > threshold], dtype=np.float64))}
            for threshold in thresholds}
