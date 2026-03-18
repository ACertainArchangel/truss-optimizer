"""Test pratt_analyse_v2 with density parameter for weight calculation"""

import io
import math
import sys
import unittest

from bridges_parametric import OnePanelPratt2D
from pratt_analyse_v2 import make_and_report


class TestWeightDisplay(unittest.TestCase):
    """Test that weight is correctly calculated and displayed in analysis."""

    def setUp(self):
        """Create a test bridge."""

        self.bridge = OnePanelPratt2D(
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

    def test_weight_display_with_steel(self):
        """Test weight calculation with steel density."""

        captured_output = io.StringIO()

        sys.stdout = captured_output

        make_and_report(self.bridge, density=7850)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        self.assertIn("Volume:", output)

        self.assertIn("Mass:", output)

        self.assertIn("Weight:", output)

        self.assertIn("Load/Weight:", output)

    def test_weight_display_with_balsa(self):
        """Test weight calculation with balsa wood density."""

        captured_output = io.StringIO()

        sys.stdout = captured_output

        make_and_report(self.bridge, density=160)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        self.assertIn("Volume:", output)

        self.assertIn("Mass:", output)

        self.assertIn("Weight:", output)

        self.assertIn("Load/Weight:", output)

    def test_no_weight_without_density(self):
        """Test that weight is not displayed when density is not provided."""

        captured_output = io.StringIO()

        sys.stdout = captured_output

        make_and_report(self.bridge, density=None)

        sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        self.assertNotIn("Volume:", output)

        self.assertNotIn("Mass:", output)

        self.assertNotIn("Weight:", output)

        self.assertIn("BRIDGE FAILURE ANALYSIS", output)


if __name__ == "__main__":

    unittest.main()
