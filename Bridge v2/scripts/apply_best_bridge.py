"""Apply best bridge parameters from best_bridge.json to manual_bridge_tester.py.
THIS WILL OVERWRITE YOUR PREVIOUS THING SO DO NOOOOOOOT USE LIGHTLY"""

import argparse
import json
import math
import re
import sys


def meters_to_inches(meters):
    """Convert meters to inches"""
    return meters / 0.0254


def load_best_bridge():
    """Load the best bridge parameters from best_bridge.json"""

    try:
        with open("best_bridge.json", "r") as f:
            params = json.load(f)
        print(" Loaded best_bridge.json")
        return params
    except FileNotFoundError:
        print("ERROR: best_bridge.json not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in best_bridge.json: {e}")
        sys.exit(1)


def display_bridge_comparison(params):
    """Display the bridge parameters that will be written"""

    print("\n" + "=" * 67)
    print("BEST BRIDGE PARAMETERS (from best_bridge.json)")
    print("=" * 67)
    print(f"\nGeometry:")
    print(f"Angle:  {math.degrees(params['angle']):.6f}°")

    print(f"Height: {meters_to_inches(params['height']):.6f} in")

    print(f"Length: {meters_to_inches(params['length']):.6f} in")

    print(f"\nThicknesses:")

    print(f"Incline:   {meters_to_inches(params['incline_thickness']):.6f} in")

    print(f"Diagonal:  {meters_to_inches(params['diagonal_thickness']):.6f} in")

    print(f"Mid Vert:  {meters_to_inches(params['mid_vert_thickness']):.6f} in")

    print(f"Side Vert: {meters_to_inches(params['side_vert_thickness']):.6f} in")

    print(f"Top:       {meters_to_inches(params['top_thickness']):.6f} in")

    print(f"Bottom:    {meters_to_inches(params['bottom_thickness']):.6f} in")

    print(f"\nDepths:")

    print(f"Incline:   {meters_to_inches(params['incline_depth']):.6f} in")

    print(f"Diagonal:  {meters_to_inches(params['diagonal_depth']):.6f} in")

    print(f"Mid Vert:  {meters_to_inches(params['mid_vert_depth']):.6f} in")

    print(f"Side Vert: {meters_to_inches(params['side_vert_depth']):.6f} in")

    print(f"Top:       {meters_to_inches(params['top_depth']):.6f} in")

    print(f"Bottom:    {meters_to_inches(params['bottom_depth']):.6f} in")

    print("=" * 67)


def confirm_overwrite(skip_confirmation=False):
    """Ask user to confirm the overwrite operation"""
    if skip_confirmation:  # ARE YOU SURE ABOUT THAT? *Cue John Cena crashing through the wall*
        print("\n Confirmation skipped (--yes flag provided). Proceeding with update...")
        return True

    print("\n WARNING: This will OVERWRITE the current bridge in manual_bridge_tester.py!")
    print("Make sure you have a backup if you want to keep the current configuration.")
    print("\nType 'yes' to confirm and proceed: ", end="")
    response = input().strip().lower()
    if response == "yes":
        print(" Confirmed. Proceeding with update...")
        return True
    else:
        print(" Operation cancelled. No changes were made.")
        return False


def update_manual_bridge_tester(params):
    """Update manual_bridge_tester.py with best bridge parameters using regex"""
    print("\n" + "=" * 67)
    print("UPDATING manual_bridge_tester.py")
    print("=" * 67)
    try:
        with open("manual_bridge_tester.py", "r") as f:
            content = f.read()
    except FileNotFoundError:
        print("ERROR: manual_bridge_tester.py not found!")
        sys.exit(1)

    angle_deg = math.degrees(params["angle"])
    height_in = meters_to_inches(params["height"])
    length_in = meters_to_inches(params["length"])
    incline_thickness_in = meters_to_inches(params["incline_thickness"])
    diagonal_thickness_in = meters_to_inches(params["diagonal_thickness"])
    mid_vert_thickness_in = meters_to_inches(params["mid_vert_thickness"])
    side_vert_thickness_in = meters_to_inches(params["side_vert_thickness"])
    top_thickness_in = meters_to_inches(params["top_thickness"])
    bottom_thickness_in = meters_to_inches(params["bottom_thickness"])
    incline_depth_in = meters_to_inches(params["incline_depth"])
    diagonal_depth_in = meters_to_inches(params["diagonal_depth"])
    mid_vert_depth_in = meters_to_inches(params["mid_vert_depth"])
    side_vert_depth_in = meters_to_inches(params["side_vert_depth"])
    top_depth_in = meters_to_inches(params["top_depth"])
    bottom_depth_in = meters_to_inches(params["bottom_depth"])
    replacements = {
        r"angle = math\.radians\([^)]+\)": f"angle = math.radians({angle_deg:.6f})",
        r"height = inches_to_meters\([^)]+\)": f"height = inches_to_meters({height_in:.6f})",
        r"length = inches_to_meters\([^)]+\)": f"length = inches_to_meters({length_in:.6f})",
        r"incline_thickness = inches_to_meters\([^)]+\)": f"incline_thickness = inches_to_meters({incline_thickness_in:.6f})",
        r"diagonal_thickness = inches_to_meters\([^)]+\)": f"diagonal_thickness = inches_to_meters({diagonal_thickness_in:.6f})",
        r"mid_vert_thickness = inches_to_meters\([^)]+\)": f"mid_vert_thickness = inches_to_meters({mid_vert_thickness_in:.6f})",
        r"side_vert_thickness = inches_to_meters\([^)]+\)": f"side_vert_thickness = inches_to_meters({side_vert_thickness_in:.6f})",
        r"top_thickness = inches_to_meters\([^)]+\)": f"top_thickness = inches_to_meters({top_thickness_in:.6f})",
        r"bottom_thickness = inches_to_meters\([^)]+\)": f"bottom_thickness = inches_to_meters({bottom_thickness_in:.6f})",
        r"incline_depth = inches_to_meters\([^)]+\)": f"incline_depth = inches_to_meters({incline_depth_in:.6f})",
        r"diagonal_depth = inches_to_meters\([^)]+\)": f"diagonal_depth = inches_to_meters({diagonal_depth_in:.6f})",
        r"mid_vert_depth = inches_to_meters\([^)]+\)": f"mid_vert_depth = inches_to_meters({mid_vert_depth_in:.6f})",
        r"side_vert_depth = inches_to_meters\([^)]+\)": f"side_vert_depth = inches_to_meters({side_vert_depth_in:.6f})",
        r"top_depth = inches_to_meters\([^)]+\)": f"top_depth = inches_to_meters({top_depth_in:.6f})",
        r"bottom_depth = inches_to_meters\([^)]+\)": f"bottom_depth = inches_to_meters({bottom_depth_in:.6f})",
    }

    updated_count = 0

    for pattern, replacement in replacements.items():
        matches = re.findall(pattern, content)
        if matches:
            content = re.sub(pattern, replacement, content)
            param_name = pattern.split("=")[0].replace("\\", "").strip().replace("r'", "")
            print(f" Updated: {param_name}")
            updated_count += 1
        else:
            param_name = pattern.split("=")[0].replace("\\", "").strip().replace("r'", "")
            print(f" Warning: Pattern not found for {param_name}")

    try:
        with open("manual_bridge_tester.py", "w") as f:
            f.write(content)
        print(f"\n Successfully updated {updated_count} parameters in manual_bridge_tester.py")
    except Exception as e:
        print(f"\nERROR: Failed to write manual_bridge_tester.py: {e}")
        sys.exit(1)


def main():
    """Main execution flow"""

    parser = argparse.ArgumentParser(
        description="Apply best bridge parameters to manual_bridge_tester.py"
    )

    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt and proceed immediately"
    )

    args = parser.parse_args()

    print("=" * 67)
    print("APPLY BEST BRIDGE TO manual_bridge_tester.py")
    print("=" * 67)

    params = load_best_bridge()

    display_bridge_comparison(params)
    if not confirm_overwrite(skip_confirmation=args.yes):
        sys.exit(0)

    update_manual_bridge_tester(params)
    print("\n" + "=" * 67)
    print("COMPLETE!")
    print("=" * 67)
    print("\nThe bridge parameters from best_bridge.json have been applied.")
    print("You can now run: python manual_bridge_tester.py")
    print()


if __name__ == "__main__":
    main()
