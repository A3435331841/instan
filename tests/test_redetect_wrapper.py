import unittest

import numpy as np

from scripts.run_sutrack_b224_redetect import (
    bfov_tuple,
    circular_delta,
    crop_wrap,
)


class RedetectWrapperTest(unittest.TestCase):
    def test_circular_delta_respects_seam(self):
        self.assertAlmostEqual(circular_delta(1.0, 1439.0, 1440.0), 2.0)
        self.assertAlmostEqual(circular_delta(1439.0, 1.0, 1440.0), -2.0)

    def test_crop_wrap_preserves_requested_width(self):
        frame = np.arange(2 * 5 * 3, dtype=np.uint8).reshape(2, 5, 3)
        crop = crop_wrap(frame, [4.0, 0.0, 3.0, 2.0])
        self.assertEqual(crop.shape, (2, 3, 3))
        np.testing.assert_array_equal(crop[:, 0], frame[:, 4])
        np.testing.assert_array_equal(crop[:, 1], frame[:, 0])
        np.testing.assert_array_equal(crop[:, 2], frame[:, 1])

    def test_bfov_tuple_is_indexable(self):
        value = bfov_tuple([0.0, 0.0, 20.0, 20.0], 1440, 720)
        self.assertEqual(len(value), 4)
        self.assertTrue(all(np.isfinite(value)))


if __name__ == "__main__":
    unittest.main()
