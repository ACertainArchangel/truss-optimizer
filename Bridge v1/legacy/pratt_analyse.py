import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from bridges_parametric import OnePanelPratt2D


def make_and_report(bridge: OnePanelPratt2D = None):
    b = (
        OnePanelPratt2D(
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
        if bridge is None
        else bridge
    )
    print("\n" + "=" * 60)
    print("  BRIDGE FAILURE ANALYSIS")
    print("=" * 60 + "\n")
    failure_modes = b.get_failure_mode_dict()
    member_results = {}
    member_types = {}
    for name, member in b.members.items():
        member_types[name] = "Tension" if isinstance(member, b.TensionMember) else "Compression"
        member_failures = []
        for mode_name, F_failure in failure_modes.items():
            if F_failure is not None and mode_name != "torsion_failure":
                if mode_name.startswith(name + "_") or mode_name == name:
                    member_failures.append(F_failure)
        if member_failures:
            member_results[name] = min(member_failures)
        else:
            member_results[name] = None
    for name in b.members.keys():
        applied_load = member_results.get(name)
        member_type = member_types[name]
        if applied_load is not None:
            print(f"  {name:.<20} {member_type:12} {applied_load:>10.2f} N")
        else:
            print(f"  {name:.<20} {member_type:12} {'ERROR':>10}")
    if member_results:
        valid_loads = [v for v in member_results.values() if v is not None]
        if valid_loads:
            min_load = min(valid_loads)
            governing = [k for k, v in member_results.items() if v == min_load][0]
            print("\n" + "-" * 60)
            print(f"  Governing Member: {governing}")
            print(f"  Critical Load:    {min_load:.2f} N")
            print("=" * 60 + "\n")
    return member_results


if __name__ == "__main__":
    make_and_report()
