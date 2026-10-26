"""Targeted analysis checks, using synthetic arrays only."""

import unittest
import numpy as np
from src.error_analysis import depth_bin_indices, delta_transitions, tail_statistics


class ErrorAnalysisChecks(unittest.TestCase):
    def test_half_open_depth_bins(self):
        values = np.array([.001, 9.999, 10., 19.999, 20., 40., 60., 99.999])
        np.testing.assert_array_equal(depth_bin_indices(values), [0, 0, 1, 1, 2, 3, 4, 4])
        for value in (0., 100., np.nan):
            with self.assertRaises(ValueError):
                depth_bin_indices(np.array([value]))

    def test_delta_losses_and_gains_strict_threshold(self):
        transitions = delta_transitions(np.array([10., 10., 10., 10.]),
                                       np.array([10., 10., 20., 12.5]),
                                       np.array([10., 20., 10., 12.5]))
        self.assertEqual([int(v.sum()) for v in transitions.values()], [1, 1, 1, 1])
        self.assertTrue(np.all(sum(v.astype(int) for v in transitions.values()) == 1))

    def test_absolute_error_tails_and_squared_error(self):
        result = tail_statistics(np.ones(4), np.array([1., 51., 101., 102.]))
        self.assertEqual(result[50.]["pixels"], 2)
        self.assertEqual(result[100.]["pixels"], 1)
        self.assertEqual(result[50.]["squared_error_sum_mm2"], 100**2 + 101**2)


if __name__ == "__main__":
    unittest.main()
