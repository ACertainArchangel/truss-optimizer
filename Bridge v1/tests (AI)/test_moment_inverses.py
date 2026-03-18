"""Tests for moment-inverse round-trips on ``OnePanelPratt2D``.

For each member that produces a bending moment, the bridge model also
provides an inverse function that maps a moment back to the applied load F.
These tests verify that applying the forward function and then the inverse
recovers the original F to within floating-point precision.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges_parametric import OnePanelPratt2D


class TestMomentInverses(unittest.TestCase):
    """Verify F → M → F round-trips are exact for moment-inverse functions.

    Only the top chord and bottom chord are tested because they are the
    members for which the model exposes an invertible moment function.  If a
    member's inverse is None it is skipped with a printed note.
    """

    def setUp(self):
        """Create a standard bridge for moment / inverse-moment testing."""

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

    def test_moment_inverses(self):
        """Apply moment function then inverse; recovered F must match original.

        For each chord member the test picks the governing moment function
        (index 0 for top chord, index 1 for bottom chord) and verifies that:

            inverse_fn( moment_fn(F) ) ≈ F

        at two representative load levels (100 N and 1234.5 N).
        """

        moments_dict = self.b.get_moments_from_F_dict()

        inverses_dict = self.b.get_required_F_from_moments_dict()

        test_members = ["top_chord", "bottom_chord"]

        for member in test_members:

            moment_fns = moments_dict[member]

            inverse_fn = inverses_dict[member]

            if inverse_fn is None:
                # Member does not provide an analytical inverse — skip.
                print(f"{member}: (skipped - no invertible function)")

                continue

            # top chord: use the first (primary) moment function;
            # bottom chord: use the second.
            idx = 0 if member == "top_chord" else 1

            moment_fn = moment_fns[idx]

            for F in (100.0, 1234.5):

                M = moment_fn(F)

                F_back = inverse_fn(M)

                self.assertTrue(
                    math.isclose(F, F_back, rel_tol=1e-6, abs_tol=1e-9),
                    msg=f"{member}[{idx}]: F={F}, M={M}, F_back={F_back}",
                )

                print(f"{member}[{idx}]: F={F}, M={M}, F_back={F_back}")


if __name__ == "__main__":

    unittest.main()
