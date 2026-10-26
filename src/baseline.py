"""Constant depth fitted only on development GT; fitting is never automatic."""

from dataclasses import dataclass
import numpy as np

from .c3vd_dataset import C3VDDataset, DEPTH_MAX_MM, ENCODED_MAX


@dataclass(frozen=True)
class ConstantMedianDepth:
    median_mm: float
    fitted_sequence: str = "trans_t1_a"

    def __post_init__(self):
        if self.fitted_sequence != "trans_t1_a":
            raise ValueError("Baseline may only be fitted on development sequence trans_t1_a")
        if not np.isfinite(self.median_mm) or self.median_mm <= 0:
            raise ValueError("Baseline depth must be finite and positive, in mm")

    @classmethod
    def fit_development(cls, dataset: C3VDDataset):
        """Exact pooled pixel median using a uint16 histogram, constant memory.

        Every selected pixel has equal weight (not a median of frame medians).
        This intentionally scans development GT only when explicitly invoked.
        """
        if dataset.sequence != "trans_t1_a":
            raise ValueError("Refusing test GT: only trans_t1_a can fit the baseline")
        histogram = np.zeros(ENCODED_MAX + 1, dtype=np.int64)
        for index in range(len(dataset)):
            raw, _, mask = dataset.load_depth(index)
            histogram += np.bincount(raw[mask], minlength=ENCODED_MAX + 1)
        count = int(histogram.sum())
        if count == 0:
            raise ValueError("No valid development depth pixels")
        cumulative = np.cumsum(histogram)
        lower = np.searchsorted(cumulative, (count - 1) // 2, side="right")
        upper = np.searchsorted(cumulative, count // 2, side="right")
        median_mm = float((lower + upper) / 2 * DEPTH_MAX_MM / ENCODED_MAX)
        return cls(median_mm)

    def predict(self, shape: tuple[int, int]) -> np.ndarray:
        if len(shape) != 2 or any(not isinstance(v, (int, np.integer)) or v <= 0 for v in shape):
            raise ValueError("Expected positive (height, width)")
        return np.full(shape, self.median_mm, dtype=np.float32)
