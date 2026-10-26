"""Pixel-wise depth metrics; both inputs must use millimeters.

Inputs are validated on the caller-selected pixels. Nonfinite values outside
the explicit mask are allowed; invalid selected values raise instead of hiding
prediction failures or silently changing the evaluation population.
"""

import numpy as np


def _selected(gt_mm, prediction_mm, valid_mask):
    gt = np.asarray(gt_mm)
    pred = np.asarray(prediction_mm)
    mask = np.asarray(valid_mask)
    if gt.ndim != 2 or gt.shape != pred.shape or gt.shape != mask.shape:
        raise ValueError("GT, prediction and mask must have the same 2D shape")
    if mask.dtype != np.bool_:
        raise ValueError("valid_mask must be boolean")
    if not np.issubdtype(gt.dtype, np.number) or not np.issubdtype(pred.dtype, np.number):
        raise ValueError("Depth arrays must be numeric")
    if np.iscomplexobj(gt) or np.iscomplexobj(pred):
        raise ValueError("Depth arrays must be real")
    if not mask.any():
        raise ValueError("No valid pixels selected")
    gt, pred = gt[mask].astype(np.float64), pred[mask].astype(np.float64)
    if not (np.isfinite(gt).all() and np.isfinite(pred).all()):
        raise ValueError("Nonfinite depth on selected pixels")
    if (gt <= 0).any() or (pred <= 0).any():
        raise ValueError("Selected GT and predicted depths must be strictly positive")
    return gt, pred


def abs_rel(gt_mm, prediction_mm, valid_mask) -> float:
    """Mean |prediction - GT| / GT (dimensionless)."""
    gt, pred = _selected(gt_mm, prediction_mm, valid_mask)
    return float(np.mean(np.abs(pred - gt) / gt))


def rmse(gt_mm, prediction_mm, valid_mask) -> float:
    """Square root of mean squared error, in mm."""
    gt, pred = _selected(gt_mm, prediction_mm, valid_mask)
    return float(np.sqrt(np.mean(np.square(pred - gt))))


def delta_1_25(gt_mm, prediction_mm, valid_mask) -> float:
    """Fraction with max(prediction/GT, GT/prediction) strictly below 1.25."""
    gt, pred = _selected(gt_mm, prediction_mm, valid_mask)
    return float(np.mean(np.maximum(pred / gt, gt / pred) < 1.25))


class DepthMetricAccumulator:
    """Pool sufficient statistics over pixels, never average frame RMSEs."""

    def __init__(self):
        self.valid_pixels = 0
        self.abs_rel_sum = 0.0
        self.squared_error_sum_mm2 = 0.0
        self.delta_1_25_count = 0

    def update(self, gt_mm, prediction_mm, valid_mask) -> dict:
        gt, pred = _selected(gt_mm, prediction_mm, valid_mask)
        count = int(gt.size)
        absolute_relative = float(np.sum(np.abs(pred - gt) / gt, dtype=np.float64))
        squared_error = float(np.sum(np.square(pred - gt), dtype=np.float64))
        delta_count = int(np.count_nonzero(np.maximum(pred / gt, gt / pred) < 1.25))
        self.valid_pixels += count
        self.abs_rel_sum += absolute_relative
        self.squared_error_sum_mm2 += squared_error
        self.delta_1_25_count += delta_count
        return {"valid_pixels": count, "abs_rel": absolute_relative / count,
                "rmse_mm": float(np.sqrt(squared_error / count)),
                "delta_1_25": delta_count / count}

    def result(self) -> dict:
        if self.valid_pixels == 0:
            raise ValueError("Cannot report an empty evaluation")
        return {"valid_pixels": self.valid_pixels,
                "abs_rel": self.abs_rel_sum / self.valid_pixels,
                "rmse_mm": float(np.sqrt(self.squared_error_sum_mm2 / self.valid_pixels)),
                "delta_1_25": self.delta_1_25_count / self.valid_pixels,
                "abs_rel_sum": self.abs_rel_sum,
                "squared_error_sum_mm2": self.squared_error_sum_mm2,
                "delta_1_25_count": self.delta_1_25_count}
