"""Tests for axial-force / required-load inverse round-trips on
``OnePanelPratt2D``.

Every bridge member has two related functions:
- ``get_axial_forces_from_F_dict``: maps applied load F → member axial force.
- ``get_required_F_from_axial_forces_dict``: the inverse, maps axial force → F.

These tests verify that applying the forward function and then the inverse
recovers the original F to within floating-point precision.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bridges_parametric import OnePanelPratt2D


class TestPrattLoads(unittest.TestCase):
    """Verify F → axial → F round-trips for all bridge members.

    A deliberately large incline thickness is used so that buckling does
    not truncate the axial-force range, ensuring all inverse functions are
    well-conditioned across the tested load range.
    """

    def setUp(self):
        """Create a bridge with a stiff incline member for clean inverse tests."""

        self.b = OnePanelPratt2D(
            angle=math.radians(30),
            height=5,
            length=10,
            incline_thickness=0.09,
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

    def test_inverse_loads(self):
        """Apply axial function then its inverse; recovered F must match original.

        For every member that appears in both dictionaries the test checks:

            load_dict[m]( axial_dict[m](F) ) ≈ F

        at two representative load levels (100 N and 1234.5 N).  This confirms
        that the inverse functions are consistent with the forward functions and
        are numerically well-behaved.
        """

        axial_dict = self.b.get_axial_forces_from_F_dict()

        load_dict = self.b.get_required_F_from_axial_forces_dict()

        # Only test members present in both dicts to guard against partial implementations.
        members = set(axial_dict.keys()) & set(load_dict.keys())

        self.assertTrue(len(members) > 0, "No common members found between axial and load dicts")

        for member in sorted(members):

            for F in (100.0, 1234.5):

                axial = axial_dict[member](F)

                F_back = load_dict[member](axial)

                self.assertTrue(
                    math.isclose(F, F_back, rel_tol=1e-6, abs_tol=1e-9),
                    msg=f"Member {member}: original F={F}, after round-trip F_back={F_back}",
                )

                print(
                    f"Member {member}: original F={F}, after round-trip F_back={F_back} (after axial={axial})"
                )


if __name__ == "__main__":

    unittest.main()
