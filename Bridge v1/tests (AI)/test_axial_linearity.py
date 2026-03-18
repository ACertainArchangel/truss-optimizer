"""Test that axial force functions are linear w.r.t. applied load."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridges_parametric import OnePanelPratt2D


class TestAxialForceLinearity(unittest.TestCase):
    """Test that axial force relationships are perfectly linear."""

    def setUp(self):
        """Create a test bridge."""

        self.bridge = OnePanelPratt2D(
            angle=math.radians(30),
            height=2.0,
            length=10.0,
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

        self.axial_dict = self.bridge.get_axial_forces_from_F_dict()

        self.member_names = [
            "incline",
            "diagonal",
            "top_chord",
            "bottom_chord",
            "mid_vert",
            "side_vert",
        ]

    def test_all_members_are_linear(self):
        """Test that all member axial forces scale linearly with applied load."""

        F_values = [1.0, 10.0, 100.0, 1000.0]

        for name in self.member_names:

            with self.subTest(member=name):

                axial_fn = self.axial_dict[name]

                coeffs = [axial_fn(F) / F for F in F_values]

                coeff_mean = sum(coeffs) / len(coeffs)

                coeff_std = (sum((c - coeff_mean) ** 2 for c in coeffs) / len(coeffs)) ** 0.5

                rel_std = coeff_std / coeff_mean if coeff_mean != 0 else 0

                self.assertLess(
                    rel_std, 1e-10, f"{name}: Axial force not linear. Coefficients vary: {coeffs}"
                )

    def test_specific_member_coefficients(self):
        """Test specific known coefficient values for key members."""

        F_test = 100.0

        incline_coeff = self.axial_dict["incline"](F_test) / F_test

        self.assertAlmostEqual(
            incline_coeff, 1.0, places=6, msg="Incline should carry full applied load"
        )

        diagonal_coeff = self.axial_dict["diagonal"](F_test) / F_test

        self.assertGreater(diagonal_coeff, 0)

        self.assertLess(diagonal_coeff, 1.0, msg="Diagonal coefficient should be between 0 and 1")

        mid_vert_coeff = self.axial_dict["mid_vert"](F_test) / F_test

        self.assertGreater(mid_vert_coeff, 0)

        self.assertLess(mid_vert_coeff, 0.5, msg="Mid vertical should carry a fraction of the load")

    def test_linearity_at_extreme_loads(self):
        """Test linearity holds at very small and very large loads."""

        for name in self.member_names:

            with self.subTest(member=name):

                axial_fn = self.axial_dict[name]

                F_small = 0.001

                F_large = 1e6

                F_mid = 100.0

                coeff_small = axial_fn(F_small) / F_small

                coeff_mid = axial_fn(F_mid) / F_mid

                coeff_large = axial_fn(F_large) / F_large

                self.assertAlmostEqual(coeff_small, coeff_mid, places=10)

                self.assertAlmostEqual(coeff_mid, coeff_large, places=10)


if __name__ == "__main__":

    suite = unittest.TestLoader().loadTestsFromTestCase(TestAxialForceLinearity)

    runner = unittest.TextTestRunner(verbosity=2)

    result = runner.run(suite)

    if result.wasSuccessful():

        print("\n" + "=" * 70)

        print("All axial force relationships are perfectly linear!")

        print("=" * 70)

        bridge = OnePanelPratt2D(
            angle=math.radians(30),
            height=2.0,
            length=10.0,
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

        axial_dict = bridge.get_axial_forces_from_F_dict()

        print("\nAxial force coefficients (F_axial = coeff × F_applied):")

        for name in ["incline", "diagonal", "top_chord", "bottom_chord", "mid_vert", "side_vert"]:

            coeff = axial_dict[name](100.0) / 100.0

            print(f"  {name:12} → {coeff:.6f}")
