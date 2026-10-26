"""Global log-depth least squares, using streaming float64 centered moments."""

from dataclasses import dataclass
import numpy as np
from .metrics import _selected


@dataclass(frozen=True)
class LogAffineCalibration:
    a: float
    b: float

    def __post_init__(self):
        if not np.isfinite(self.a) or not np.isfinite(self.b):
            raise ValueError("Log-affine parameters must be finite")

    def apply(self, original_mm: np.ndarray) -> np.ndarray:
        original = np.asarray(original_mm, dtype=np.float64)
        if original.ndim != 2 or not np.isfinite(original).all() or (original <= 0).any():
            raise ValueError("Log-affine input must be finite positive 2D depth in mm")
        with np.errstate(over="raise", under="raise", invalid="raise"):
            corrected = np.exp(self.a + self.b * np.log(original))
        if not np.isfinite(corrected).all() or (corrected <= 0).any():
            raise ValueError("Invalid log-affine output; no clipping or silent filtering is allowed")
        return corrected


class StreamingLogLeastSquares:
    """Each valid pixel has equal weight; merge centered moments frame by frame."""

    def __init__(self):
        self.n = 0
        self.mean_x = self.mean_y = 0.0
        self.sxx = self.sxy = self.syy = 0.0

    def update(self, gt_mm, prediction_mm, mask):
        gt, pred = _selected(gt_mm, prediction_mm, mask)
        x, y = np.log(pred), np.log(gt)
        n = int(x.size)
        mx, my = float(x.mean()), float(y.mean())
        xc, yc = x - mx, y - my
        xx, xy, yy = float(np.dot(xc, xc)), float(np.dot(xc, yc)), float(np.dot(yc, yc))
        total = self.n + n
        dx, dy = mx - self.mean_x, my - self.mean_y
        weight = self.n * n / total
        self.sxx += xx + dx * dx * weight
        self.sxy += xy + dx * dy * weight
        self.syy += yy + dy * dy * weight
        self.mean_x += dx * n / total
        self.mean_y += dy * n / total
        self.n = total

    def fit_development(self, *, sequence: str) -> LogAffineCalibration:
        if sequence != "trans_t1_a":
            raise ValueError("Only development split trans_t1_a may fit log-affine parameters")
        if self.n < 2 or not np.isfinite([self.mean_x, self.mean_y, self.sxx, self.sxy, self.syy]).all():
            raise ValueError("Insufficient or nonfinite log-depth statistics")
        tolerance = np.finfo(np.float64).eps * max(1., self.mean_x ** 2) * self.n
        if self.sxx <= tolerance:
            raise ValueError("Prediction log-depth has zero or numerically negligible variance; slope is undefined")
        b = self.sxy / self.sxx
        a = self.mean_y - b * self.mean_x
        return LogAffineCalibration(float(a), float(b))

    def diagnostics(self, calibration: LogAffineCalibration) -> dict:
        residual_sse = self.syy - 2 * calibration.b * self.sxy + calibration.b ** 2 * self.sxx
        return {"valid_pixels": self.n, "mean_log_prediction": self.mean_x, "mean_log_gt": self.mean_y,
                "centered_sum_xx": self.sxx, "centered_sum_xy": self.sxy, "centered_sum_yy": self.syy,
                "fitted_log_sse": float(residual_sse), "fitted_log_mse": float(residual_sse / self.n)}


def transformed_quantiles_inplace(raw_pool, transform, *, decreasing=False) -> dict:
    """Exact P05/P50/P95 via in-place partition of a NEW disk-backed raw pool.

    Transform both adjacent order statistics BEFORE linear interpolation.
    This also handles negative slopes, and avoids assuming nonlinear functions
    commute with interpolated quantiles. Never pass an original prediction file.
    """
    probabilities = np.array([.05, .5, .95], dtype=np.float64)
    positions = probabilities * (raw_pool.size - 1)
    lower, upper = np.floor(positions).astype(np.int64), np.ceil(positions).astype(np.int64)
    ranks = np.concatenate([lower, upper])
    if decreasing:
        ranks = raw_pool.size - 1 - ranks
    raw_pool.partition(np.unique(ranks))
    values = np.asarray(raw_pool[ranks], dtype=np.float64).reshape(1, -1)
    transformed = transform(values).ravel()
    quantiles = transformed[:3] + (transformed[3:] - transformed[:3]) * (positions - lower)
    return dict(zip(("p05", "median", "p95"), map(float, quantiles)))
