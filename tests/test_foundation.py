"""Small synthetic checks only; never reads the real development/test GT."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from src.baseline import ConstantMedianDepth
from src.c3vd_dataset import C3VDDataset, decode_depth_mm, make_valid_mask
from src.metrics import abs_rel, rmse, delta_1_25


class FoundationChecks(unittest.TestCase):
    def test_known_metrics_and_strict_threshold(self):
        gt = np.array([[10., 20., 40., 80.]])
        pred = np.array([[10., 25., 20., 100.]])
        mask = np.ones_like(gt, dtype=bool)
        self.assertAlmostEqual(abs_rel(gt, pred, mask), 0.25)
        self.assertAlmostEqual(rmse(gt, pred, mask), np.sqrt(825 / 4))
        self.assertEqual(delta_1_25(gt, pred, mask), 0.25)

    def test_mask_excludes_only_explicit_pixels(self):
        gt = np.array([[10., np.nan]])
        pred = np.array([[12., np.inf]])
        mask = np.array([[True, False]])
        self.assertAlmostEqual(abs_rel(gt, pred, mask), 0.2)
        self.assertEqual(rmse(gt, pred, mask), 2.)

    def test_invalid_selected_inputs_raise(self):
        gt = np.ones((1, 2))
        for metric in (abs_rel, rmse, delta_1_25):
            for pred in (np.array([[1., 0.]]), np.array([[1., np.nan]]),
                         np.array([[1., -2.]])):
                with self.subTest(metric=metric.__name__, pred=pred):
                    with self.assertRaises(ValueError):
                        metric(gt, pred, np.ones_like(gt, dtype=bool))
            with self.assertRaises(ValueError):
                metric(gt, gt, np.zeros_like(gt, dtype=bool))
            with self.assertRaises(ValueError):
                metric(gt, gt, np.ones_like(gt))
            with self.assertRaises(ValueError):
                metric(gt, np.ones((2, 1)), np.ones_like(gt, dtype=bool))
            with self.assertRaises(ValueError):
                metric(np.zeros_like(gt), gt, np.ones_like(gt, dtype=bool))

    def test_uint16_endpoints_and_mask_policy(self):
        raw = np.array([[0, 32768, 65535]], dtype=np.uint16)
        depth = decode_depth_mm(raw)
        self.assertEqual(depth.dtype, np.float32)
        np.testing.assert_allclose(depth, [[0., 32768 * 100 / 65535, 100.]])
        self.assertTrue(make_valid_mask(depth).all())
        np.testing.assert_array_equal(make_valid_mask(depth, lambda d: d > 0), [[False, True, True]])
        with self.assertRaises(ValueError):
            decode_depth_mm(raw.astype(np.uint8))
        with self.assertRaises(ValueError):
            make_valid_mask(depth, lambda d: np.ones_like(d))

    @staticmethod
    def write_pair(directory, frame_id, raw):
        Image.fromarray(np.zeros((*raw.shape, 3), dtype=np.uint8)).save(directory / f"{frame_id}_color.png")
        Image.fromarray(raw).save(directory / f"{frame_id}_depth.tiff")

    def test_pairing_lossless_depth_and_unpaired_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "trans_t1_a" / "trans_t1_a"
            directory.mkdir(parents=True)
            raw = np.array([[0, 256, 65535]], dtype=np.uint16)
            self.write_pair(directory, "0010", raw)
            self.write_pair(directory, "0002", raw)
            dataset = C3VDDataset(temporary, "trans_t1_a")
            self.assertEqual([p.frame_id for p in dataset.pairs], ["0002", "0010"])
            np.testing.assert_array_equal(dataset[0].depth_raw, raw)
            self.assertEqual(dataset[0].rgb.shape, (1, 3, 3))
            Image.fromarray(raw).save(directory / "0099_depth.tiff")
            with self.assertRaises(ValueError):
                C3VDDataset(temporary, "trans_t1_a")

    def test_exact_global_median_and_test_leakage_guard(self):
        # Unequal frame sizes detect the incorrect median-of-frame-medians method.
        for arrays in ([np.array([[10]], dtype=np.uint16),
                        np.array([[20, 30, 40]], dtype=np.uint16)],
                       [np.array([[10, 20, 30]], dtype=np.uint16)]):
            class SyntheticDevelopment:
                sequence = "trans_t1_a"

                def __len__(self):
                    return len(arrays)

                def load_depth(self, index):
                    raw = arrays[index]
                    return raw, decode_depth_mm(raw), np.ones_like(raw, dtype=bool)

            baseline = ConstantMedianDepth.fit_development(SyntheticDevelopment())
            expected = np.median(np.concatenate([a.ravel() for a in arrays])) * 100 / 65535
            self.assertAlmostEqual(baseline.median_mm, expected)
            np.testing.assert_allclose(baseline.predict((2, 3)), expected)

        class ForbiddenTest:
            sequence = "trans_t2_b"

            def __len__(self):
                raise AssertionError("Test data must never be accessed")

        with self.assertRaises(ValueError):
            ConstantMedianDepth.fit_development(ForbiddenTest())
        with self.assertRaises(ValueError):
            ConstantMedianDepth(1, fitted_sequence="trans_t2_b")


if __name__ == "__main__":
    unittest.main()
