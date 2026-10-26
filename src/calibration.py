"""One predeclared calibration: development pooled median GT / prediction."""

from dataclasses import dataclass
import numpy as np

from .metrics import _selected

QUANTILE_PROBABILITIES = np.array([0., .05, .25, .5, .75, .95, 1.])
QUANTILE_NAMES = ("min", "p05", "p25", "median", "p75", "p95", "max")


def histogram_depth_distribution(histogram: np.ndarray) -> dict:
    """Exact pooled GT quantiles, linear order-statistic interpolation, in mm.

    Compute from lossless uint16 codes and decode in float64, as in baseline
    fitting. Metric GT remains the existing float32 decoded array.
    """
    histogram = np.asarray(histogram)
    if histogram.shape != (65536,) or (histogram < 0).any() or histogram.sum() == 0:
        raise ValueError("Expected nonempty nonnegative uint16 histogram")
    count = int(histogram.sum())
    cumulative = histogram.cumsum()
    positions = QUANTILE_PROBABILITIES * (count - 1)
    lower = np.searchsorted(cumulative, np.floor(positions).astype(np.int64), side="right")
    upper = np.searchsorted(cumulative, np.ceil(positions).astype(np.int64), side="right")
    values = (lower + (upper - lower) * (positions - np.floor(positions))) * (100. / 65535)
    codes_mm = np.arange(65536, dtype=np.float64) * (100. / 65535)
    mean = float(np.dot(histogram, codes_mm) / count)
    variance = float(np.dot(histogram, (codes_mm - mean) ** 2) / count)
    return {"valid_pixels": count, "quantiles_mm": dict(zip(QUANTILE_NAMES, map(float, values))),
            "mean_mm": mean, "std_mm": float(np.sqrt(variance)),
            "definition": "lossless encoded GT, float64 code * 100 / 65535"}


def pooled_prediction_distribution(valid_frame_predictions) -> dict:
    """Exact quantiles of every valid prediction pixel, not frame medians."""
    pooled = np.concatenate([np.asarray(array).ravel() for array in valid_frame_predictions])
    if pooled.size == 0 or not np.isfinite(pooled).all() or (pooled <= 0).any():
        raise ValueError("Pooled predictions must be nonempty, finite and positive")
    count = int(pooled.size)
    mean = float(np.mean(pooled, dtype=np.float64))
    std = float(np.std(pooled, dtype=np.float64))
    values = np.quantile(pooled, QUANTILE_PROBABILITIES, method="linear", overwrite_input=True)
    return {"valid_pixels": count, "quantiles_mm": dict(zip(QUANTILE_NAMES, map(float, values))),
            "mean_mm": mean, "std_mm": std,
            "definition": "exact linear quantiles pooled over masked float32 predictions"}


def pearson_structure(gt_mm, prediction_mm, valid_mask) -> dict:
    """Per-frame Pearson over fixed GT-valid pixels; constants are undefined.

    No p-values or independence assumptions: pixels are spatially correlated.
    This describes depth association, not learned causality or metric accuracy.
    """
    gt, pred = _selected(gt_mm, prediction_mm, valid_mask)
    gt_constant, pred_constant = bool(np.ptp(gt) == 0), bool(np.ptp(pred) == 0)
    gt_centered, pred_centered = gt - gt.mean(), pred - pred.mean()
    gt_ss, pred_ss = float(np.dot(gt_centered, gt_centered)), float(np.dot(pred_centered, pred_centered))
    if gt_constant or pred_constant:
        reason = "zero_variance_both" if gt_constant and pred_constant else (
            "zero_variance_gt" if gt_constant else "zero_variance_prediction")
        r = None
    else:
        r = float(np.clip(np.dot(gt_centered, pred_centered) / np.sqrt(gt_ss * pred_ss), -1, 1))
        reason = "defined"
    return {"valid_pixels": int(gt.size), "pearson_r": r, "status": reason,
            "gt_std_mm": float(np.sqrt(gt_ss / gt.size)),
            "prediction_std_mm": float(np.sqrt(pred_ss / pred.size))}


def summarize_structure(rows) -> dict:
    values = np.array([row["pearson_r"] for row in rows if row["pearson_r"] is not None], dtype=np.float64)
    return {"method": "per-frame Pearson on c3vd-interior-v1 valid pixels",
            "aggregation": "unweighted descriptive summary over defined frames",
            "constant_variance_policy": "undefined correlation is null, excluded from r summary, counted explicitly",
            "frames": len(rows), "defined_frames": int(values.size), "undefined_frames": len(rows) - int(values.size),
            "mean_r": float(values.mean()) if values.size else None,
            "median_r": float(np.median(values)) if values.size else None,
            "min_r": float(values.min()) if values.size else None,
            "max_r": float(values.max()) if values.size else None,
            "positive_frames": int(np.count_nonzero(values > 0)),
            "negative_frames": int(np.count_nonzero(values < 0)),
            "zero_correlation_frames": int(np.count_nonzero(values == 0)),
            "interpretation_limit": "association only; not a significance test, causality claim or replacement for depth metrics"}


@dataclass(frozen=True)
class GlobalScaleCalibration:
    scale_factor: float

    def __post_init__(self):
        if not np.isfinite(self.scale_factor) or self.scale_factor <= 0:
            raise ValueError("Scale factor must be finite and positive")

    @classmethod
    def fit_development(cls, gt_median_mm: float, prediction_median_mm: float, *, sequence: str):
        if sequence != "trans_t1_a":
            raise ValueError("Only trans_t1_a may fit the scale")
        if not np.isfinite(gt_median_mm) or gt_median_mm <= 0:
            raise ValueError("GT median must be finite and positive")
        if not np.isfinite(prediction_median_mm) or prediction_median_mm <= 0:
            raise ValueError("Prediction median must be finite and positive")
        return cls(float(gt_median_mm / prediction_median_mm))

    def apply(self, original_mm: np.ndarray) -> np.ndarray:
        original_mm = np.asarray(original_mm)
        if original_mm.ndim != 2 or not np.isfinite(original_mm).all() or (original_mm <= 0).any():
            raise ValueError("Expected positive finite 2D depth")
        calibrated = (original_mm.astype(np.float64) * self.scale_factor).astype(np.float32)
        if not np.isfinite(calibrated).all() or (calibrated <= 0).any():
            raise ValueError("Invalid calibrated depth after float32 conversion")
        return calibrated
