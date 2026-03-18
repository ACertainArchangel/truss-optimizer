"""
Unit tests to ensure ``pratt_analyse.py`` (v1) and ``pratt_analyse_v2.py``
produce identical per-member minimum failure loads for identical bridge
geometries.  Both scripts must return the same dictionary keys and values
(to 6 decimal places) so that either can be used interchangeably.
"""

import math
import unittest

import legacy.pratt_analyse as pratt_analyse
import pratt_analyse_v2
from bridges_parametric import OnePanelPratt2D


class TestAnalyseParity(unittest.TestCase):
    """Test that v1 and v2 analysis scripts produce identical failure-load results.

    Two bridge configurations are tested:
    - Default: 30° incline, 2 m height, 10 m span, 20 mm × 50 mm sections.
    - Custom:  45° incline, 1.5 m height, 8 m span, 15 mm × 40 mm sections.
    """

    def test_default_bridge_parity(self):
        """Test that both scripts produce identical results for the default bridge."""

        bridge = OnePanelPratt2D(
            angle=math.radians(30),
            height=2,
            length=10,
            incline_thickness=0.02,
            diagonal_thickness=0.02,
            mid_vert_thickness=0.02,
            side_vert_thickness=0.02,
            top_thickness=0.02,
            bottom_thickness=0.02,
            incline_depth=0.05,
            diagonal_depth=0.05,
            mid_vert_depth=0.05,
            side_vert_depth=0.05,
            top_depth=0.05,
            bottom_depth=0.05,
            E=200e9,
            sigma_compression=250e6,
            sigma_tension=400e6,
        )

        results_v1 = pratt_analyse.make_and_report(bridge)

        results_v2 = pratt_analyse_v2.make_and_report(bridge)

        self.assertEqual(
            set(results_v1.keys()),
            set(results_v2.keys()),
            "Both scripts should analyze the same members",
        )

        for member_name in results_v1.keys():

            val1 = results_v1[member_name]

            val2 = results_v2[member_name]

            if val1 is None and val2 is None:

                continue

            self.assertIsNotNone(val1, f"v1 returned None for {member_name}")

            self.assertIsNotNone(val2, f"v2 returned None for {member_name}")

            self.assertAlmostEqual(
                val1,
                val2,
                places=6,
                msg=f"Mismatch for {member_name}: v1={val1:.6f}, v2={val2:.6f}",
            )

    def test_custom_bridge_parity(self):
        """Test with a custom bridge configuration."""

        bridge = OnePanelPratt2D(
            angle=math.radians(45),
            height=1.5,
            length=8,
            incline_thickness=0.015,
            diagonal_thickness=0.015,
            mid_vert_thickness=0.015,
            side_vert_thickness=0.015,
            top_thickness=0.015,
            bottom_thickness=0.015,
            incline_depth=0.04,
            diagonal_depth=0.04,
            mid_vert_depth=0.04,
            side_vert_depth=0.04,
            top_depth=0.04,
            bottom_depth=0.04,
            E=200e9,
            sigma_compression=250e6,
            sigma_tension=400e6,
        )

        results_v1 = pratt_analyse.make_and_report(bridge)

        results_v2 = pratt_analyse_v2.make_and_report(bridge)

        self.assertEqual(set(results_v1.keys()), set(results_v2.keys()))

        for member_name in results_v1.keys():

            val1 = results_v1[member_name]

            val2 = results_v2[member_name]

            if val1 is None and val2 is None:

                continue

            self.assertIsNotNone(val1)

            self.assertIsNotNone(val2)

            self.assertAlmostEqual(
                val1,
                val2,
                places=6,
                msg=f"Mismatch for {member_name}: v1={val1:.6f}, v2={val2:.6f}",
            )


if __name__ == "__main__":

    unittest.main()
