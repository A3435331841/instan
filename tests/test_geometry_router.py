import unittest

from scripts.run_geometry_routed_b224_t224 import route_adaptive_b224, route_t224


class GeometryRouterTest(unittest.TestCase):
    def test_compact_nonpolar_uses_fast_expert(self):
        self.assertEqual(route_t224((0.0, 10.0, 12.0, 25.0))[0], True)

    def test_tiny_extreme_pole_stays_on_b224(self):
        self.assertEqual(route_t224((0.0, -87.0, 4.0, 5.0))[0], False)

    def test_medium_fov_default_is_b224(self):
        self.assertEqual(route_t224((0.0, 0.0, 45.0, 60.0))[0], False)

    def test_compact_high_latitude_uses_adaptive_b224(self):
        self.assertEqual(route_adaptive_b224((0.0, -80.0, 25.0, 30.0))[0], True)

    def test_moderate_fov_uses_adaptive_b224(self):
        self.assertEqual(route_adaptive_b224((0.0, 0.0, 42.0, 66.0))[0], True)

    def test_narrow_normal_view_keeps_default_b224(self):
        self.assertEqual(route_adaptive_b224((0.0, -25.0, 23.5, 48.5))[0], False)

    def test_wide_view_keeps_safe_b224_policy(self):
        self.assertEqual(route_adaptive_b224((0.0, 20.0, 129.0, 168.0))[0], False)


if __name__ == "__main__":
    unittest.main()
