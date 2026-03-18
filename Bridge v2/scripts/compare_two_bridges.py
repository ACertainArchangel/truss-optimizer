"""
Compare two bridge designs from manual_bridge_tester.py files using voodoo regex extraction.

Usage:
    python scripts/compare_two_bridges.py [--unit imperial|metric]
"""

import argparse
import math
import re
import sys


def extract_bridge_params(filepath):
    """Extract bridge parameters from manual_bridge_tester.py file using regex"""

    with open(filepath, "r") as f:
        content = f.read()
    params = {}
    angle_match = re.search(r"angle\s*=\s*math\.radians\s*\(\s*([\d.]+)\s*\)", content)

    if angle_match:
        params["angle"] = float(angle_match.group(1))

    height_match = re.search(r"height\s*=\s*inches_to_meters\s*\(\s*([\d.]+)\s*\)", content)

    if height_match:
        params["height"] = float(height_match.group(1))

    length_match = re.search(r"length\s*=\s*inches_to_meters\s*\(\s*([\d.]+)\s*\)", content)

    if length_match:
        params["length"] = float(length_match.group(1))

    thickness_patterns = {
        "incline_thickness": r"incline_thickness\s*=\s*inches_to_meters\s*\(\s*([\d.]+)\s*\)",
        "diagonal_thickness": r"diagonal_thickness\s*=\s*inches_to_meters\s*\(\s*([\d.]+)\s*\)",
        "mid_vert_thickness": r"mid_vert_thickness\s*=\s*inches_to_meters\s*\(\s*([\d.]+)\s*\)",
        "side_vert_thickness": r"side_vert_thickness\s*=\s*inches_to_meters\s*\(\s*([\d.]+)\s*\)",
        "top_thickness": r"top_thickness\s*=\s*inches_to_meters\s*\(\s*([\d.]+)\s*\)",
        "bottom_thickness": r"bottom_thickness\s*=\s*inches_to_meters\s*\(\s*([\d.]+)\s*\)",
    }

    for param_name, pattern in thickness_patterns.items():
        match = re.search(pattern, content)

        if match:
            params[param_name] = float(match.group(1))

    depth_names = [
        "incline_depth",
        "diagonal_depth",
        "mid_vert_depth",
        "side_vert_depth",
        "top_depth",
        "bottom_depth",
    ]

    for param_name in depth_names:

        pattern_decimal = rf"{param_name}\s*=\s*inches_to_meters\s*\(\s*([\d.]+)\s*\)"
        match = re.search(pattern_decimal, content)

        if match:
            params[param_name] = float(match.group(1))
        else:
            pattern_fraction = rf"{param_name}\s*=\s*inches_to_meters\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)"
            match = re.search(pattern_fraction, content)
            if match:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                params[param_name] = numerator / denominator

    return params


def main():

    parser = argparse.ArgumentParser(description="Compare two bridge designs")

    parser.add_argument(
        "--unit",
        choices=["imperial", "metric"],
        default="imperial",
        help="Unit system for display (default: imperial)",
    )

    args = parser.parse_args()

    try:
        old = extract_bridge_params("manual_bridge_tester copy.py")
        print(" Loaded OLD bridge from 'manual_bridge_tester copy.py'")
    except FileNotFoundError:
        print(" Could not find 'manual_bridge_tester copy.py'")
        sys.exit(1)

    except Exception as e:
        print(f" Error loading OLD bridge: {e}")
        sys.exit(1)
    try:
        new = extract_bridge_params("manual_bridge_tester.py")
        print(" Loaded NEW bridge from 'manual_bridge_tester.py'")

    except FileNotFoundError:
        print(" Could not find 'manual_bridge_tester.py'")
        sys.exit(1)

    except Exception as e:
        print(f" Error loading NEW bridge: {e}")
        sys.exit(1)

    print()

    def format_dimension(inches, unit_system):
        if unit_system == "imperial":
            return f"{inches:.6f} in"
        else:
            cm = inches * 2.54
            return f"{cm:.6f} cm"

    def format_diff(diff_inches, unit_system):
        if unit_system == "imperial":
            return f"{diff_inches:+.6f} in"
        else:
            diff_cm = diff_inches * 2.54
            return f"{diff_cm:+.6f} cm"

    print("=" * 100)
    print("BRIDGE PARAMETER COMPARISON")
    print("=" * 100)
    print(f"\nUnits: {args.unit.upper()}")
    print("\nOLD = manual_bridge_tester copy.py")
    print("NEW = manual_bridge_tester.py")
    print()
    print(f"{'Parameter':<25} {'OLD':<25} {'NEW':<25} {'Difference':<25} {'%'}")
    print("-" * 105)

    param_order = [
        "angle",
        "height",
        "length",
        "incline_thickness",
        "diagonal_thickness",
        "mid_vert_thickness",
        "side_vert_thickness",
        "top_thickness",
        "bottom_thickness",
        "incline_depth",
        "diagonal_depth",
        "mid_vert_depth",
        "side_vert_depth",
        "top_depth",
        "bottom_depth",
    ]

    for key in param_order:
        if key not in old or key not in new:
            continue

        old_val = old[key]
        new_val = new[key]
        diff = new_val - old_val

        if key == "angle":
            old_str = f"{old_val:.6f}°"
            new_str = f"{new_val:.6f}°"
            diff_str = f"{diff:+.6f}°"
            pct = (diff / old_val * 100) if old_val != 0 else 0
        elif key == "length":
            old_str = format_dimension(old_val, args.unit)
            new_str = format_dimension(new_val, args.unit)
            diff_str = format_diff(diff, args.unit)
            pct = (diff / old_val * 100) if old_val != 0 else 0
        else:
            old_str = format_dimension(old_val, args.unit)
            new_str = format_dimension(new_val, args.unit)
            diff_str = format_diff(diff, args.unit)
            pct = (diff / old_val * 100) if old_val != 0 else 0

        print(f"{key:<25} {old_str:<25} {new_str:<25} {diff_str:<25} {pct:+.2f}%")

    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)
    print()
    print("Key differences:")

    significant_changes = []

    for key in param_order:
        if key not in old or key not in new:
            continue

        old_val = old[key]
        new_val = new[key]
        pct_change = ((new_val - old_val) / old_val * 100) if old_val != 0 else 0

        if abs(pct_change) > 1.0:
            significant_changes.append((key, pct_change, new_val - old_val))

    if significant_changes:
        for key, pct, diff in sorted(significant_changes, key=lambda x: abs(x[1]), reverse=True):
            symbol = "↑" if diff > 0 else "↓"
            print(f"  {symbol} {key:<23} {pct:+6.2f}%")
    else:
        print("No significant changes (>1%) detected")

    print()
    print("Note: To see performance metrics, run the bridges through analysis:")
    print(". python manual_bridge_tester.py")
    print("  python 'manual_bridge_tester copy.py'")
    print()


if __name__ == "__main__":
    main()
