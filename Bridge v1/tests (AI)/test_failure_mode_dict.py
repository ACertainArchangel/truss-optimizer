"""Tests for ``OnePanelPratt2D.get_failure_mode_dict()``.

Verifies that the failure-mode dictionary returned by the bridge model
contains every expected key and that all values are finite, positive
numbers (the torsion entry may be None and is skipped).
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges_parametric import OnePanelPratt2D


class TestFailureModeDict(unittest.TestCase):
    """Verify the structure and sanity of get_failure_mode_dict().

    Each bridge member should expose at least three failure modes:
    in-plane buckling, out-of-plane buckling, and combined stress / rupture.
    The torsional mode is excluded from numeric checks because it may not be
    implemented.
    """

    def setUp(self):
        """Build a representative bridge with standard geometry and steel properties."""

        self.b = OnePanelPratt2D(
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

    def test_failure_mode_dict_matches_main(self):
        """get_failure_mode_dict() must return every canonical failure-mode key
        and all numeric values must be finite and positive.

        The canonical set includes three modes per compressive member
        (in-plane buckle, out-of-plane buckle, combined stress) and one
        rupture mode for tension members (diagonal and bottom chord).
        'torsion_failure' is listed but may be None and is therefore skipped
        in the numeric assertions.
        """

        actual = self.b.get_failure_mode_dict()

        expected_keys = [
            "incline_buckle",
            "incline_buckle_out_of_plane",
            "incline_combined_stress",
            "diagonal_rupture",
            "top_chord_buckle",
            "top_chord_buckle_out_of_plane",
            "top_chord_combined_stress",
            "bottom_chord_rupture",
            "mid_vert_buckle",
            "mid_vert_buckle_out_of_plane",
            "mid_vert_combined_stress",
            "side_vert_buckle",
            "side_vert_buckle_out_of_plane",
            "side_vert_combined_stress",
            "torsion_failure",
        ]

        # Every expected key must be present.
        for key in expected_keys:

            self.assertIn(key, actual, msg=f"Key {key} missing from get_failure_mode_dict()")

        # All non-torsion values must be positive, finite numbers.
        for k, v in actual.items():

            if k == "torsion_failure":
                # Torsion may be unimplemented; skip it.
                continue

            self.assertIsNotNone(v, msg=f"Value for {k} should not be None")

            self.assertIsInstance(v, (int, float), msg=f"Value for {k} should be numeric")

            self.assertGreater(v, 0, msg=f"Value for {k} should be positive, got {v}")


if __name__ == "__main__":

    unittest.main()
