import unittest

from scripts.run_geometry_routed_b224_t224 import (
    route_adaptive_b224,
    route_fixed_b224,
    route_noswitch_b224,
    route_t224,
)


class GeometryRouterTest(unittest.TestCase):
    def test_compact_nonpolar_uses_fast_expert(self):
        self.assertEqual(route_t224((0.0, 10.0, 5.0, 5.0))[0], True)

    def test_medium_narrow_band_uses_b224_exploration(self):
        self.assertEqual(route_t224((0.0, 10.0, 12.0, 12.0))[0], False)

    def test_narrow_vertical_risk_band_keeps_b224(self):
        self.assertEqual(route_t224((0.0, 10.0, 12.0, 25.0))[0], False)

    def test_compact_rescue_band_uses_no_switch_b224(self):
        self.assertEqual(route_t224((0.0, -8.0, 5.9, 20.0))[0], False)
        self.assertEqual(route_noswitch_b224((0.0, -8.0, 5.9, 20.0))[0], True)

    def test_fixed_compact_vertical_band_is_geometry_only(self):
        self.assertEqual(route_fixed_b224((0.0, -12.6, 11.8, 14.0))[0], True)
        self.assertEqual(route_fixed_b224((0.0, -66.0, 11.8, 14.0))[0], False)

    def test_fixed_tiny_band_excludes_polar_views(self):
        self.assertEqual(route_fixed_b224((0.0, -26.6, 5.8, 3.0))[0], True)
        self.assertEqual(route_fixed_b224((0.0, -49.0, 2.5, 4.8))[0], False)

    def test_fixed_compact_envelope_keeps_polar_views_out(self):
        self.assertEqual(route_fixed_b224((0.0, -7.5, 5.9, 20.3))[0], True)
        self.assertEqual(route_fixed_b224((0.0, 16.4, 5.6, 16.0))[0], True)
        self.assertEqual(route_fixed_b224((0.0, -49.0, 2.5, 4.8))[0], False)

    def test_tiny_nonpolar_uses_no_switch_b224(self):
        self.assertEqual(route_noswitch_b224((0.0, -49.0, 2.5, 4.8))[0], True)

    def test_high_lat_compact_vertical_view_keeps_existing_route(self):
        self.assertEqual(route_noswitch_b224((0.0, -55.0, 11.6, 14.8))[0], False)

    def test_medium_vertical_risk_view_keeps_adaptive_route(self):
        self.assertEqual(route_noswitch_b224((0.0, -12.6, 11.8, 14.0))[0], False)

    def test_extreme_pole_stays_on_fast_expert(self):
        self.assertEqual(route_noswitch_b224((0.0, -80.0, 5.5, 5.2))[0], False)

    def test_tiny_extreme_pole_stays_on_b224(self):
        self.assertEqual(route_t224((0.0, -87.0, 4.0, 5.0))[0], False)

    def test_medium_fov_default_is_b224(self):
        self.assertEqual(route_t224((0.0, 0.0, 45.0, 60.0))[0], False)

    def test_compact_high_latitude_uses_adaptive_b224(self):
        self.assertEqual(route_adaptive_b224((0.0, -80.0, 25.0, 30.0))[0], True)

    def test_moderate_fov_uses_adaptive_b224(self):
        self.assertEqual(route_adaptive_b224((0.0, 0.0, 42.0, 66.0))[0], True)

    def test_high_latitude_safety_band_keeps_default_b224(self):
        self.assertEqual(route_adaptive_b224((0.0, -79.0, 30.3, 30.4))[0], False)

    def test_narrow_normal_view_keeps_default_b224(self):
        self.assertEqual(route_adaptive_b224((0.0, -25.0, 23.5, 48.5))[0], False)

    def test_wide_view_keeps_safe_b224_policy(self):
        self.assertEqual(route_adaptive_b224((0.0, 20.0, 129.0, 168.0))[0], False)


if __name__ == "__main__":
    unittest.main()
