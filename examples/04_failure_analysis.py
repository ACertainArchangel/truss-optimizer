"""
Example 4: Failure Mode Analysis

This example demonstrates how to perform detailed structural
analysis on a bridge design.
"""

import math
from truss_optimizer import PrattTruss, FailureAnalyzer, materials
from truss_optimizer.utils import inches_to_meters


def main():
    print("=" * 70)
    print("  Structural Failure Analysis")
    print("=" * 70)
    
    # Create a bridge design
    # These parameters might come from optimization or manual design
    bridge = PrattTruss(
        span=inches_to_meters(18.5),
        height=inches_to_meters(6.0),
        angle=math.radians(30),
        
        # Member dimensions
        incline_thickness=inches_to_meters(0.25),
        incline_depth=inches_to_meters(0.25),
        diagonal_thickness=inches_to_meters(0.1875),  # 3/16"
        diagonal_depth=inches_to_meters(0.25),
        mid_vert_thickness=inches_to_meters(0.1875),
        mid_vert_depth=inches_to_meters(0.25),
        side_vert_thickness=inches_to_meters(0.1875),
        side_vert_depth=inches_to_meters(0.25),
        top_thickness=inches_to_meters(0.25),
        top_depth=inches_to_meters(0.375),   # 3/8"
        bottom_thickness=inches_to_meters(0.25),
        bottom_depth=inches_to_meters(0.375),
        
        material=materials.BalsaWood()
    )
    
    # Create analyzer
    analyzer = FailureAnalyzer(bridge)
    
    # Print the full report
    analyzer.print_report(unit_system='metric')
    
    # You can also get specific information programmatically
    print("\n  PROGRAMMATIC ACCESS:")
    print("  " + "-" * 40)
    print(f"  Critical Load: {analyzer.critical_load:.2f} N")
    print(f"  Governing Mode: {analyzer.governing_mode}")
    
    # Get safety factors
    print(f"\n  Safety Factors (relative to governing mode):")
    sf = analyzer.safety_factors
    sorted_sf = sorted(sf.items(), key=lambda x: x[1])
    for mode, factor in sorted_sf[:6]:
        status = "◀ CRITICAL" if factor == 1.0 else ""
        print(f"    {mode:<35} SF = {factor:.2f} {status}")
    
    # Get summary by member
    print(f"\n  Summary by Member:")
    summary = analyzer.get_member_summary()
    for member, modes in summary.items():
        min_load = min(modes.values())
        print(f"    {member}: min load = {min_load:.1f} N")
    
    # Save the report
    analyzer.save_report("failure_analysis.txt")
    print(f"\n  Report saved to: failure_analysis.txt")


if __name__ == "__main__":
    main()
