"""Targeted tests for streaming least squares and the new transform only."""

import unittest
import numpy as np
from src.contrast_calibration import LogAffineCalibration, StreamingLogLeastSquares, transformed_quantiles_inplace


class ContrastCalibrationChecks(unittest.TestCase):
    def test_streamed_fit_matches_direct_lstsq_with_unequal_batches(self):
        prediction = np.array([[1., 2., 3., 5., 8., 13., 21.]])
        gt = np.array([[2., 3., 3.5, 6., 11., 15., 30.]])
        streaming = StreamingLogLeastSquares()
        for begin, end in ((0, 1), (1, 3), (3, 7)):
            streaming.update(gt[:, begin:end], prediction[:, begin:end], np.ones((1, end-begin), dtype=bool))
        fitted = streaming.fit_development(sequence="trans_t1_a")
        direct = np.linalg.lstsq(np.column_stack([np.ones(prediction.size), np.log(prediction.ravel())]),
                                 np.log(gt.ravel()), rcond=None)[0]
        np.testing.assert_allclose([fitted.a, fitted.b], direct, rtol=1e-13, atol=1e-13)
        with self.assertRaises(ValueError):
            streaming.fit_development(sequence="trans_t2_b")

    def test_known_parameters_and_mask(self):
        prediction = np.array([[2., 4., 8., np.nan]])
        gt = np.array([[3., 12., 48., np.nan]])
        streaming = StreamingLogLeastSquares()
        streaming.update(gt, prediction, np.array([[True, True, True, False]]))
        fitted = streaming.fit_development(sequence="trans_t1_a")
        np.testing.assert_allclose([fitted.a, fitted.b], [np.log(.75), 2.], atol=1e-13)
        self.assertEqual(streaming.n, 3)

    def test_constant_prediction_is_explicit_failure(self):
        streaming = StreamingLogLeastSquares()
        streaming.update(np.array([[1., 2., 3.]]), np.ones((1, 3)), np.ones((1, 3), dtype=bool))
        with self.assertRaises(ValueError):
            streaming.fit_development(sequence="trans_t1_a")

    def test_no_clipping_and_invalid_inputs_raise(self):
        fitted = LogAffineCalibration(0., 2.)
        np.testing.assert_allclose(fitted.apply(np.array([[20.]])), [[400.]])
        self.assertEqual(fitted.apply(np.array([[20.]])).dtype, np.float64)
        for value in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                fitted.apply(np.array([[value]]))
        with self.assertRaises(FloatingPointError):
            LogAffineCalibration(1000., 1.).apply(np.ones((1, 1)))

    def test_exact_nonlinear_quantiles_with_both_slope_signs(self):
        raw = np.array([2., 4., 8., 16., 32., 64.], dtype=np.float32)
        for b in (2., -2., 0.):
            fitted = LogAffineCalibration(.3, b)
            result = transformed_quantiles_inplace(raw.copy(), fitted.apply, decreasing=b < 0)
            expected = np.quantile(fitted.apply(raw.reshape(1, -1)), [.05, .5, .95])
            np.testing.assert_allclose(list(result.values()), expected, rtol=1e-14)


if __name__ == "__main__":
    unittest.main()
