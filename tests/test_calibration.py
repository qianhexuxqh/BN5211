"""Small synthetic checks for the single prespecified calibration method."""

import unittest
import numpy as np
from src.calibration import (GlobalScaleCalibration, histogram_depth_distribution,
    pooled_prediction_distribution, pearson_structure, summarize_structure)


class CalibrationChecks(unittest.TestCase):
    def test_pooled_prediction_median_not_median_of_frame_medians(self):
        distribution = pooled_prediction_distribution([np.array([1.], dtype=np.float32),
                                                       np.array([10., 20., 30.], dtype=np.float32)])
        self.assertEqual(distribution["quantiles_mm"]["median"], 15.)
        self.assertNotEqual(distribution["quantiles_mm"]["median"], 10.5)

    def test_gt_histogram_exact_odd_and_even(self):
        for codes in ([10, 20, 30], [10, 20, 30, 40]):
            histogram = np.bincount(codes, minlength=65536)
            result = histogram_depth_distribution(histogram)
            self.assertAlmostEqual(result["quantiles_mm"]["median"], np.median(codes) * 100 / 65535)

    def test_unique_global_scale_and_no_clipping(self):
        calibration = GlobalScaleCalibration.fit_development(20., 400., sequence="trans_t1_a")
        self.assertEqual(calibration.scale_factor, .05)
        np.testing.assert_allclose(calibration.apply(np.array([[400., 4000.]], dtype=np.float32)), [[20., 200.]])
        with self.assertRaises(ValueError):
            GlobalScaleCalibration.fit_development(20., 400., sequence="trans_t2_b")
        with self.assertRaises(ValueError):
            GlobalScaleCalibration.fit_development(20., 0., sequence="trans_t1_a")
        with self.assertRaises(ValueError):
            calibration.apply(np.array([[np.nan]]))

    def test_pearson_and_constant_variance(self):
        gt = np.array([[1., 2., 3.]])
        mask = np.ones_like(gt, dtype=bool)
        self.assertAlmostEqual(pearson_structure(gt, gt * 10, mask)["pearson_r"], 1.)
        self.assertAlmostEqual(pearson_structure(gt, 4 - gt, mask)["pearson_r"], -1.)
        undefined = pearson_structure(gt, np.ones_like(gt), mask)
        self.assertIsNone(undefined["pearson_r"])
        self.assertEqual(undefined["status"], "zero_variance_prediction")
        summary = summarize_structure([undefined, pearson_structure(gt, gt, mask)])
        self.assertEqual(summary["undefined_frames"], 1)
        self.assertEqual(summary["defined_frames"], 1)


if __name__ == "__main__":
    unittest.main()
