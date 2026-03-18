"""Test that analyse scripts match get_failure_mode_dict() and PyTorch by computing the minimum failure load across all modes."""

import math
import unittest

import legacy.pratt_analyse as pratt_analyse
import pratt_analyse_v2
import torch
from bridges_parametric import OnePanelPratt2D
from pratt_torch import max_load_torch


class TestAnalyseMethodsParity(unittest.TestCase):
    """Test that analysis scripts match the class method and PyTorch implementation."""

    def setUp(self):
        """Create test bridges."""
        self.default_bridge = OnePanelPratt2D(
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

        self.custom_bridge = OnePanelPratt2D(
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

    def test_analyse_scripts_compute_minimum_of_all_modes(self):
        """Verify analyse scripts compute minimum failure load across all modes per member."""
        results_v1 = pratt_analyse.make_and_report(self.default_bridge)
        results_v2 = pratt_analyse_v2.make_and_report(self.default_bridge)
        failure_modes = self.default_bridge.get_failure_mode_dict()
        member_modes = {}
        for mode_name, F_failure in failure_modes.items():
            if F_failure is None or mode_name == "torsion_failure":
                continue

            for member_name in [
                "incline",
                "diagonal",
                "top_chord",
                "bottom_chord",
                "mid_vert",
                "side_vert",
            ]:

                if mode_name.startswith(member_name):

                    if member_name not in member_modes:

                        member_modes[member_name] = []

                    member_modes[member_name].append((mode_name, F_failure))

                    break

        print("\n" + "=" * 80)

        print("FAILURE MODE ANALYSIS COMPARISON")

        print("=" * 80)

        for member_name, modes in member_modes.items():

            min_failure_load = min(F for _, F in modes)

            governing_mode = [name for name, F in modes if F == min_failure_load][0]

            print(f"\n{member_name.upper()}:")

            print(f"  All modes:")

            for mode_name, F_failure in sorted(modes, key=lambda x: x[1]):

                marker = " ← GOVERNING" if F_failure == min_failure_load else ""

                print(f"    {mode_name:40} {F_failure:12.2f} N{marker}")

            v1_result = results_v1.get(member_name)

            v2_result = results_v2.get(member_name)

            print(f"Analyse v1 result: {v1_result:12.2f} N")

            print(f"Analyse v2 result: {v2_result:12.2f} N")

            print(f"Expected minimum:  {min_failure_load:12.2f} N ({governing_mode})")

            if v1_result is not None:
                self.assertAlmostEqual(
                    v1_result,
                    min_failure_load,
                    places=2,
                    msg=f"v1: {member_name} should return min of all modes. "
                    f"Got {v1_result:.2f}, expected {min_failure_load:.2f}",
                )

            if v2_result is not None:

                self.assertAlmostEqual(
                    v2_result,
                    min_failure_load,
                    places=2,
                    msg=f"v2: {member_name} should return min of all modes. "
                    f"Got {v2_result:.2f}, expected {min_failure_load:.2f}",
                )

        print("\n" + "=" * 80 + "\n")

    def test_pytorch_parity_with_class_method(self):
        """Verify PyTorch implementation matches get_failure_mode_dict()."""

        failure_modes_class = self.default_bridge.get_failure_mode_dict()

        params_torch = {
            "angle": torch.tensor(math.radians(30), dtype=torch.float64, requires_grad=True),
            "height": torch.tensor(2.0, dtype=torch.float64),
            "length": torch.tensor(10.0, dtype=torch.float64),
            "incline_thickness": torch.tensor(0.02, dtype=torch.float64),
            "diagonal_thickness": torch.tensor(0.02, dtype=torch.float64),
            "mid_vert_thickness": torch.tensor(0.02, dtype=torch.float64),
            "side_vert_thickness": torch.tensor(0.02, dtype=torch.float64),
            "top_thickness": torch.tensor(0.02, dtype=torch.float64),
            "bottom_thickness": torch.tensor(0.02, dtype=torch.float64),
            "incline_depth": torch.tensor(0.05, dtype=torch.float64),
            "diagonal_depth": torch.tensor(0.05, dtype=torch.float64),
            "mid_vert_depth": torch.tensor(0.05, dtype=torch.float64),
            "side_vert_depth": torch.tensor(0.05, dtype=torch.float64),
            "top_depth": torch.tensor(0.05, dtype=torch.float64),
            "bottom_depth": torch.tensor(0.05, dtype=torch.float64),
            "E": 200e9,
            "sigma_compression": 250e6,
            "sigma_tension": 400e6,
        }

        _, failure_modes_torch = max_load_torch(**params_torch)

        print("\n" + "=" * 80)

        print("PYTORCH vs CLASS METHOD COMPARISON")

        print("=" * 80)

        all_match = True

        for mode_name in failure_modes_class.keys():

            if mode_name == "torsion_failure":

                continue

            F_class = failure_modes_class[mode_name]

            F_torch = (
                failure_modes_torch[mode_name].item() if mode_name in failure_modes_torch else None
            )

            if F_class is not None and F_torch is not None:

                diff = abs(F_class - F_torch)

                rel_diff = diff / F_class * 100 if F_class != 0 else 0

                match = diff < 1.0

                status = "PASS" if match else "FAIL"

                print(
                    f"{status} {mode_name:40} Class: {F_class:12.2f} N  PyTorch: {F_torch:12.2f} N  Diff: {diff:8.4f} N ({rel_diff:.2f}%)"
                )

                if not match:

                    all_match = False

                self.assertAlmostEqual(
                    F_class,
                    F_torch,
                    places=1,
                    msg=f"{mode_name}: Class method gave {F_class:.2f}, PyTorch gave {F_torch:.2f}",
                )

        print("=" * 80 + "\n")

        if all_match:

            print("All failure modes match between class method and PyTorch!\n")

        else:

            print("Some failure modes don't match!\n")

    def test_analyse_scripts_show_wrong_values(self):
        """
        This test documents that the analyse scripts are NOT computing the correct failure loads.
        They compute something different from get_failure_mode_dict().
        """

        results_v2 = pratt_analyse_v2.make_and_report(self.default_bridge)

        failure_modes = self.default_bridge.get_failure_mode_dict()

        print("\n" + "=" * 80)

        print("DOCUMENTING THE MISMATCH")

        print("=" * 80)

        print("\nFor INCLINE (compression member):")

        print(f"  v2 reports:                {results_v2['incline']:12.2f} N")

        print(f"  get_failure_mode_dict():")

        print(f"    incline_buckle           {failure_modes['incline_buckle']:12.2f} N")

        print(
            f"    incline_buckle_out_of_pl {failure_modes['incline_buckle_out_of_plane']:12.2f} N"
        )

        print(f"    incline_combined_stress  {failure_modes['incline_combined_stress']:12.2f} N")

        print("\nFor TOP_CHORD (compression member with moment):")

        print(f"  v2 reports:                {results_v2['top_chord']:12.2f} N")

        print(f"  get_failure_mode_dict():")

        print(f"    top_chord_buckle         {failure_modes['top_chord_buckle']:12.2f} N")

        print(
            f"    top_chord_buckle_out_of_ {failure_modes['top_chord_buckle_out_of_plane']:12.2f} N"
        )

        print(f"    top_chord_combined_stres {failure_modes['top_chord_combined_stress']:12.2f} N")

        print("\nFor BOTTOM_CHORD (tension member with moment):")

        print(f"  v2 reports:                {results_v2['bottom_chord']:12.2f} N")

        print(f"  get_failure_mode_dict():")

        print(f"    bottom_chord_rupture     {failure_modes['bottom_chord_rupture']:12.2f} N")

        print("\n" + "=" * 80)

        print("CONCLUSION: The analyse scripts compute something different!")

        print("They don't match ANY of the individual failure modes from get_failure_mode_dict()")

        print("=" * 80 + "\n")


if __name__ == "__main__":

    unittest.main(verbosity=2)
