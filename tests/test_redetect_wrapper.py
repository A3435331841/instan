import unittest

import numpy as np

from scripts.run_sutrack_b224_redetect import (
    bfov_tuple,
    circular_delta,
    crop_wrap,
)
from scripts.run_geometry_routed_od_recovery import route_direct_od, route_recovery
from scripts.run_probe_b224 import route_factor_probe, route_probe_b224


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

    def test_recovery_gate_is_geometry_only(self):
        self.assertTrue(route_recovery((0.0, -20.0, 72.0, 140.0))[0])
        self.assertTrue(route_recovery((0.0, -56.0, 37.0, 73.0))[0])
        self.assertTrue(route_recovery((0.0, -51.0, 69.0, 83.0))[0])
        self.assertFalse(route_recovery((0.0, -35.0, 58.0, 109.0))[0])
        self.assertFalse(route_recovery((0.0, -20.0, 20.0, 30.0))[0])

    def test_direct_od_gate_is_narrow_polar_geometry(self):
        self.assertTrue(route_direct_od((0.0, -66.0, 6.3, 12.0))[0])
        self.assertFalse(route_direct_od((0.0, -70.0, 9.2, 9.0))[0])

    def test_probe_preserves_polar_and_moderate_geometry(self):
        self.assertFalse(route_probe_b224((0.0, -70.0, 36.0, 44.0))[0])
        self.assertFalse(route_probe_b224((0.0, 0.0, 27.0, 58.0))[0])
        self.assertFalse(route_probe_b224((0.0, 8.7, 9.2, 28.3))[0])
        self.assertFalse(route_probe_b224((0.0, -18.8, 45.6, 90.4))[0])
        self.assertTrue(route_probe_b224((0.0, 0.0, 22.0, 56.0))[0])

    def test_factor_probe_is_limited_to_nonpolar_large_views(self):
        self.assertTrue(route_factor_probe((0.0, -34.9, 58.1, 108.6))[0])
        self.assertTrue(route_factor_probe((0.0, -7.9, 88.4, 151.4))[0])
        self.assertFalse(route_factor_probe((0.0, -50.0, 58.0, 108.0))[0])
        self.assertFalse(route_factor_probe((0.0, -25.0, 20.0, 40.0))[0])


if __name__ == "__main__":
    unittest.main()
