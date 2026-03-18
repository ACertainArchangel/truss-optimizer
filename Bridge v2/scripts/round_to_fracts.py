"""Round depth params to nearest 1/BASE fractions and update manual_bridge_tester.py."""

import argparse
import re

import sympy


def extract_depth_params(filepath):
    """Extract depth parameters from manual_bridge_tester.py using regex"""

    with open(filepath, "r") as f:
        content = f.read()

    depths = {}

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
            depths[param_name] = float(match.group(1))
        else:
            pattern_fraction = rf"{param_name}\s*=\s*inches_to_meters\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)"
            match = re.search(pattern_fraction, content)
            if match:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                depths[param_name] = numerator / denominator

    return depths, content


def round_to_fraction(value, base):
    """Round a decimal value to the nearest 1/base fraction"""
    rounded = round(value * base)
    return int(rounded), base


def update_depths_in_content(content, depths, base):
    """Update depth parameters in the file content with fractional representations"""

    updated_content = content

    depth_names = [
        "incline_depth",
        "diagonal_depth",
        "mid_vert_depth",
        "side_vert_depth",
        "top_depth",
        "bottom_depth",
    ]

    for param_name in depth_names:
        if param_name not in depths:
            continue

        value = depths[param_name]
        numerator, denominator = round_to_fraction(value, base)
        new_line = f"{param_name} = inches_to_meters({numerator}/{denominator})"
        pattern = rf"{param_name}\s*=\s*inches_to_meters\s*\([^)]+\)"
        match = re.search(pattern, updated_content)

        if match:
            old_line = match.group(0)
            updated_content = updated_content.replace(old_line, new_line, 1)
            print(f"  {param_name:<20} {value:.6f} in → {numerator}/{denominator} in")

        else:
            print(f"   {param_name:<20} not found in file")
    return updated_content


def main():
    parser = argparse.ArgumentParser(description="Round depth parameters to fractions")

    parser.add_argument(
        "--base", type=int, default=16, help="Fraction base (default: 16 for sixteenths)"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without writing to file"
    )

    args = parser.parse_args()
    filepath = "manual_bridge_tester.py"

    print("=" * 67)
    print(f"ROUNDING DEPTHS TO 1/{args.base} FRACTIONS")
    print("=" * 67)
    print()

    try:
        depths, content = extract_depth_params(filepath)
        print(f"Extracted {len(depths)} depth parameters from '{filepath}'")
        print()

    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find '{filepath}'")
        return 1

    except Exception as e:
        print(f"FATAL ERROR: Error reading file: {e}")
        return 1

    if not depths:
        print("FATAL ERROR: No depth parameters found/")
        return 1

    print(f"Rounding to nearest 1/{args.base}:")

    print()

    updated_content = update_depths_in_content(content, depths, args.base)

    print()

    if args.dry_run:
        print("=" * 67)
        print("DRY RUN - No changes written to file")
        print("=" * 67)
        print()
        print("To apply changes, run without --dry-run flag")
    else:
        try:
            with open(filepath, "w") as f:
                f.write(updated_content)

            print("=" * 67)
            print(f"Successfully updated '{filepath}'")
            print("=" * 67)

        except Exception as e:
            print(f"FATAL ERROR: Error writing file: {e}")
            return 1

    return 0


if __name__ == "__main__":

    exit(main())
