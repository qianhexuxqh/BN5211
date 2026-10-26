"""Targeted checks for the newly added pooled evaluation protocol."""

import unittest
import numpy as np
from src.c3vd_dataset import decode_depth_mm
from src.evaluation import evaluation_mask, interior_depth_mask
from src.metrics import DepthMetricAccumulator, abs_rel, rmse, delta_1_25


class EvaluationChecks(unittest.TestCase):
    def test_raw_and_decoded_mask_equivalence_all_codes(self):
        raw = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        mask = evaluation_mask(raw)
        np.testing.assert_array_equal(mask, interior_depth_mask(decode_depth_mm(raw)))
        self.assertEqual(int(mask.sum()), 65534)
        self.assertFalse(mask[0, 0])
        self.assertFalse(mask[-1, -1])

    def test_pooled_metrics_equal_concatenated_pixels(self):
        gt_frames = [np.array([[10.]]), np.array([[20., 40., 80.]])]
        predictions = [np.array([[11.]]), np.array([[30., 20., 80.]])]
        accumulator = DepthMetricAccumulator()
        per_frame = []
        for gt, pred in zip(gt_frames, predictions):
            per_frame.append(accumulator.update(gt, pred, np.ones_like(gt, dtype=bool)))
        result = accumulator.result()
        gt, pred = np.concatenate(gt_frames, axis=1), np.concatenate(predictions, axis=1)
        mask = np.ones_like(gt, dtype=bool)
        self.assertEqual(result["valid_pixels"], 4)
        self.assertAlmostEqual(result["abs_rel"], abs_rel(gt, pred, mask))
        self.assertAlmostEqual(result["rmse_mm"], rmse(gt, pred, mask))
        self.assertAlmostEqual(result["delta_1_25"], delta_1_25(gt, pred, mask))
        self.assertNotAlmostEqual(result["rmse_mm"], np.mean([r["rmse_mm"] for r in per_frame]))

    def test_empty_accumulator_and_invalid_prediction(self):
        accumulator = DepthMetricAccumulator()
        with self.assertRaises(ValueError):
            accumulator.result()
        with self.assertRaises(ValueError):
            accumulator.update(np.ones((1, 1)), np.zeros((1, 1)), np.ones((1, 1), dtype=bool))
        self.assertEqual(accumulator.valid_pixels, 0)


if __name__ == "__main__":
    unittest.main()
