from main import connector_cost, epoxy_cost, fixed_cost

"""
Manual Bridge Tester - Define your own bridge and analyze its performance"""

import math
from math import cos, sin, tan

import matplotlib.pyplot as plt
import numpy as np
from bridges_parametric import OnePanelPratt2D
from config import K_EFFECTIVE_LENGTH
from materials import (
    Aluminum,
    ConservativeBalsaWood,
    HighDensityBalsaWood,
    LowDensityBalsaWood,
    Material,
    OchromaWithEpoxy,
    OchromaWood,
    Steel,
)
from pratt_analyse_v2 import make_and_report
from pratt_visualiser import PrattVisualiser
from utils import feet_to_meters, inches_to_meters

DISPLAY_UNITS = "imperial"

FRACTION_BASE = 16

NUM_PLANES = 2


def create_custom_bridge():
    """
    Modify the values below to test different bridge designs.
    All dimensions should be in meters (use helper functions for conversion).
    """

    material = OchromaWood()
    angle = math.radians(32.78)
    height = inches_to_meters(5.16)
    length = inches_to_meters(18.5)

    incline_thickness = inches_to_meters(0.5)

    incline_depth = inches_to_meters(8 / 16)

    diagonal_thickness = inches_to_meters(0.15934)

    diagonal_depth = inches_to_meters(2 / 16)

    mid_vert_thickness = inches_to_meters(0.500000)

    mid_vert_depth = inches_to_meters(8 / 16)

    side_vert_thickness = inches_to_meters(0.14966)

    side_vert_depth = inches_to_meters(3 / 16)

    top_thickness = inches_to_meters(0.500000)

    top_depth = inches_to_meters(8 / 16)

    bottom_thickness = inches_to_meters(0.48385)

    bottom_depth = inches_to_meters(5 / 16)

    """
    BREAK
    """

    bridge = OnePanelPratt2D(
        angle=angle,
        height=height,
        length=length,
        incline_thickness=incline_thickness,
        diagonal_thickness=diagonal_thickness,
        mid_vert_thickness=mid_vert_thickness,
        side_vert_thickness=side_vert_thickness,
        top_thickness=top_thickness,
        bottom_thickness=bottom_thickness,
        incline_depth=incline_depth,
        diagonal_depth=diagonal_depth,
        mid_vert_depth=mid_vert_depth,
        side_vert_depth=side_vert_depth,
        top_depth=top_depth,
        bottom_depth=bottom_depth,
        E=material.E,
        sigma_compression=material.sigma_compression,
        sigma_tension=material.sigma_tension,
        material=material,
        K=K_EFFECTIVE_LENGTH,
    )

    return bridge, material


def format_length(meters, unit_system="imperial"):
    """Convert meters to display units"""

    if unit_system == "imperial":
        inches = meters / 0.0254

        return f"{inches:.2f} in"

    else:
        cm = meters * 100

        return f"{cm:.2f} cm"


def format_area(sq_meters, unit_system="imperial"):
    """Convert square meters to display units"""

    if unit_system == "imperial":
        sq_inches = sq_meters / (0.0254**2)
        return f"{sq_inches:.3f} in²"
    else:
        sq_cm = sq_meters * 1e4
        return f"{sq_cm:.3f} cm²"


def format_volume(cubic_meters, unit_system="imperial"):
    """Convert cubic meters to display units"""

    if unit_system == "imperial":
        cubic_inches = cubic_meters / (0.0254**3)
        return f"{cubic_inches:.2f} in³"
    else:
        cubic_cm = cubic_meters * 1e6
        return f"{cubic_cm:.2f} cm³"


def format_mass(kg, unit_system="imperial"):
    """Convert kg to display units"""

    if unit_system == "imperial":
        ounces = kg * 35.274
        return f"{ounces:.2f} oz"

    else:
        grams = kg * 1000
        return f"{grams:.2f} g"


def format_depth_as_fraction(meters):
    """Convert meters to inches and display as fraction with denominator 16"""

    inches = meters / 0.0254
    if FRACTION_BASE == 0:
        return f"{inches:.2f} in"

    units = round(inches * FRACTION_BASE)
    return f"{units}/{FRACTION_BASE} in"


def print_bridge_summary(bridge, material, unit_system=DISPLAY_UNITS):
    """Print a summary of the bridge design parameters"""

    print("\n" + "=" * 67)
    print("  2D BRIDGE DESIGN SUMMARY")
    print("=" * 67 + "\n")
    print(f"Material: {material.__class__.__name__}")
    print(f"Young's Modulus (E):        {material.E/1e9:.2f} GPa")
    print(f"Compression Strength (σ_c): {material.sigma_compression/1e6:.2f} MPa")
    print(f"Tension Strength (σ_t):     {material.sigma_tension/1e6:.2f} MPa")
    print(f"Density:                    {material.density:.2f} kg/m³")
    print(f"\nGeometry:")
    print(f"Angle:  {math.degrees(bridge.angle):.2f}° ({bridge.angle:.4f} rad)")
    print(f"Height: {format_length(bridge.height, unit_system)}")
    print(f"Length: {format_length(bridge.length, unit_system)}")
    print(f"\nCross-Sections (in plane × out of plane):")

    for name, member in bridge.members.items():
        display_name = name.replace("_", " ").title()

        if hasattr(member, "thickness") and hasattr(member, "depth"):
            thickness = member.thickness
            depth = member.depth
            area = thickness * depth
            thickness_inches = thickness / 0.0254

            print(
                f"{display_name:15} {thickness_inches:.5f} in × "
                f"{format_depth_as_fraction(depth)} (Area: {format_area(area, unit_system)})"
            )

        elif hasattr(member, "area") or hasattr(member, "cross_sec_area"):
            area = member.area if hasattr(member, "area") else member.cross_sec_area
            depth = member.depth if hasattr(member, "depth") else None

            if depth:
                thickness = area / depth
                thickness_inches = thickness / 0.0254

                print(
                    f"{display_name:15} {thickness_inches:.5f} in × "
                    f"{format_depth_as_fraction(depth)} (Area: {format_area(area, unit_system)})"
                )
            else:
                print(f"{display_name:15} Area: {format_area(area, unit_system)}")


def print_detailed_failure_analysis(bridge, unit_system=DISPLAY_UNITS):
    """Print detailed failure analysis for all 14 failure modes"""

    print("\n" + "=" * 67)
    print("DETAILED FAILURE MODE ANALYSIS")
    print("(Applied loads on bridge that cause each failure mode)")
    print("=" * 67 + "\n")

    failure_dict = bridge.get_failure_mode_dict()

    member_groups = {
        "incline": [],
        "diagonal": [],
        "mid_vert": [],
        "side_vert": [],
        "top_chord": [],
        "bottom_chord": [],
    }

    for mode_name, F_failure in failure_dict.items():
        for member in member_groups.keys():
            if mode_name.startswith(member):
                member_groups[member].append((mode_name, F_failure))

    for member_name, modes in member_groups.items():
        if not modes:
            continue
        member_obj = bridge.members[member_name]
        member_type = "Tension" if isinstance(member_obj, bridge.TensionMember) else "Compression"
        print(f"\n{member_name.upper().replace('_', ' ')} ({member_type}):")
        print("-" * 67)

        for mode_name, F_failure in modes:
            if F_failure is not None:
                mode_display = mode_name.replace(member_name + "_", "").replace("_", " ").title()
                print(f"{mode_display:40} {F_failure:>12.2f} N (applied load)")
            else:
                mode_display = mode_name.replace(member_name + "_", "").replace("_", " ").title()
                print(f"{mode_display:40} {'N/A':>12}")
        valid_failures = [F for _, F in modes if F is not None]
        if valid_failures:
            min_failure = min(valid_failures)
            print(f"{'→ Critical applied load for this member:':40} {min_failure:>12.2f} N")


def visualize_bridge(bridge, save_filename=None):
    """Visualize the bridge geometry"""

    print("\n" + "=" * 67)
    print("BRIDGE VISUALIZATION")
    print("=" * 67 + "\n")

    vis = PrattVisualiser()

    bridge_image = vis.visualise(bridge)

    plt.figure(figsize=(12, 8))
    plt.imshow(np.array(bridge_image), cmap="gray")
    plt.title("Bridge Geometry", fontsize=16, fontweight="bold")
    plt.axis("off")

    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_filename}")

    plt.show()

    vis.shutdown()


def print_parts_list(bridge: OnePanelPratt2D, material: Material):
    """Print construction parts list for building the bridge"""

    print("\n" + "=" * 67)
    print("CONSTRUCTION PARTS LIST")
    print("=" * 67 + "\n")

    incline_thick = bridge.members["incline"].thickness / 0.0254
    incline_depth = bridge.members["incline"].depth / 0.0254
    diagonal_area = bridge.members["diagonal"].area
    diagonal_depth_m = bridge.members["diagonal"].depth
    diagonal_thick = (diagonal_area / diagonal_depth_m) / 0.0254
    diagonal_depth = diagonal_depth_m / 0.0254
    top_thick = bridge.members["top_chord"].thickness / 0.0254
    top_depth = bridge.members["top_chord"].depth / 0.0254
    bottom_area = bridge.members["bottom_chord"].area
    bottom_depth_m = bridge.members["bottom_chord"].depth
    bottom_thick = (bottom_area / bottom_depth_m) / 0.0254
    bottom_depth = bottom_depth_m / 0.0254
    mid_vert_thick = bridge.members["mid_vert"].thickness / 0.0254
    mid_vert_depth = bridge.members["mid_vert"].depth / 0.0254
    side_vert_thick = bridge.members["side_vert"].thickness / 0.0254
    side_vert_depth = bridge.members["side_vert"].depth / 0.0254
    height_in = bridge.height / 0.0254
    len_in = bridge.length / 0.0254
    pythag = lambda a, b: math.sqrt(a**2 + b**2)
    incline_length = height_in / sin(bridge.angle)
    diagonal_length = height_in / cos(bridge.phi)
    top_length = len_in - 2 * height_in / tan(bridge.angle)
    bottom_length = len_in
    vert_length = height_in - bottom_thick - top_thick
    t_lat_length = 6 - 2 * top_depth
    t_mid_length = pythag(height_in / tan(bridge.angle), 6 - 2 * top_depth)
    b_mid_length = pythag(height_in / tan(bridge.angle), 6 - 2 * bottom_depth)
    b_lat_length = 6 - 2 * bottom_depth
    b_out_length = pythag(len_in - 2 * height_in / tan(bridge.angle), 6 - 2 * bottom_depth)
    lateral_thick = 0.18750

    total_volume = sum(
        [
            4 * (incline_thick * incline_length * incline_depth),
            4 * (diagonal_thick * diagonal_length * diagonal_depth),
            2 * (top_thick * top_length * top_depth),
            2 * (bottom_thick * bottom_length * bottom_depth),
            2 * (mid_vert_thick * vert_length * mid_vert_depth),
            4 * (side_vert_thick * vert_length * side_vert_depth),
            3 * (lateral_thick * t_lat_length) * 3 / 16,
            2 * (lateral_thick * t_mid_length) * 3 / 16,
            2 * (lateral_thick * b_mid_length) * 3 / 16,
            3 * (lateral_thick * b_lat_length) * 3 / 16,
            2 * (lateral_thick * b_out_length) * 3 / 16,
        ]
    )

    print("Rectangles Cut from 1/16 in plywood:")
    print(f"{'Dimensions (in)':<25} {'Area (in²)':<12} {'Qty':<6} {'Member'}")
    print("-" * 70)

    def qty_str(multiplier, depth_in_inches):
        if FRACTION_BASE == 0:
            return f"{round(multiplier * depth_in_inches, 2):<6}"
        fraction_unit = 1.0 / float(FRACTION_BASE)
        return f"{round(multiplier * depth_in_inches / fraction_unit):<6}"
    print(
        f"{incline_thick:.5f} × {incline_length:.5f}        {incline_thick * incline_length:>6.2f}        {qty_str(4, incline_depth):<6} (Incline)"
    )
    print(
        f"{diagonal_thick:.5f} × {diagonal_length:.5f}        {diagonal_thick * diagonal_length:>6.2f}        {qty_str(4, diagonal_depth):<6} (Diagonal)"
    )
    print(
        f"{top_thick:.5f} × {top_length:.5f}        {top_thick * top_length:>6.2f}        {qty_str(2, top_depth):<6} (Top Chord)"
    )
    print(
        f"{bottom_thick:.5f} × {bottom_length:.5f}       {bottom_thick * bottom_length:>6.2f}        {qty_str(2, bottom_depth):<6} (Bottom Chord)"
    )
    print(
        f"{mid_vert_thick:.5f} × {vert_length:.5f}        {mid_vert_thick * vert_length:>6.2f}        {qty_str(2, mid_vert_depth):<6} (Mid Vert)"
    )
    print(
        f"{side_vert_thick:.5f} × {vert_length:.5f}        {side_vert_thick * vert_length:>6.2f}        {qty_str(4, side_vert_depth):<6} (Side Vert)"
    )
    print(
        f"{lateral_thick:.5f} × {t_lat_length:.5f}        {lateral_thick * t_lat_length:>6.2f}        {3*3:<6} (T-lats)"    )

    print(
        f"{lateral_thick:.5f} × {t_mid_length:.5f}        {lateral_thick * t_mid_length:>6.2f}        {2*3:<6} (T-mid/s)"
    )
    print(
        f"{lateral_thick:.5f} × {b_mid_length:.5f}        {lateral_thick * b_mid_length:>6.2f}        {2*3:<6} (B-mid/s)"
    )
    print(
        f"{lateral_thick:.5f} × {b_lat_length:.5f}        {lateral_thick * b_lat_length:>6.2f}        {3*3:<6} (B-lats)"
    )
    print(
        f"{lateral_thick:.5f} × {b_out_length:.5f}        {lateral_thick * b_out_length:>6.2f}        {2*3:<6} (B-out/s)"
    )

    print("\n\nEpoxy:")
    print("  215000 PSI High-Strength: yes")
    print(f"\n\nOptimizer assumed {fixed_cost:.2f} N fixed cost per truss plane but we")
    print("calculate the actual connector weight here. fixed_cost was meant as an")
    print("easily diferentiable proxy during optimization.")
    print(f"\n\nTotal Parts List Volume: {total_volume:.2f} in³")
    print(f"Total Volume in Metric: {total_volume * 0.0000163871:.6f} m³")

    mass_kg = total_volume * 0.0000163871 * material.density

    print(f"Total Parts List Mass: {mass_kg} kg")

    weight_N = mass_kg * 9.81

    print(f"\n\nTotal Parts List Weight (all parts listed above): {weight_N:.2f} N")

    per_plane_material_weight = bridge.get_total_volume() * material.density * 9.81
    assumed_build_weight = NUM_PLANES * (fixed_cost + per_plane_material_weight)
    total_build_weight = weight_N + (NUM_PLANES * epoxy_cost)
    per_plane_total_weight = total_build_weight / NUM_PLANES

    print(
        f"Per-plane material weight (no connectors - from geometry): {per_plane_material_weight:.2f} N"
    )

    print(f"Per-plane total weight (derived from parts-list): {per_plane_total_weight:.2f} N")

    print(
        f"Total build weight (parts list + epoxy for {NUM_PLANES} planes): {total_build_weight:.2f} N"
    )

    failure_dict = bridge.get_failure_mode_dict()

    critical_load = min([v for v in failure_dict.values() if v is not None])

    double_load = NUM_PLANES * critical_load

    print(f"\nDouble Original Supportable Force: {double_load:.2f} N")

    print(f"Expected Score (assumed fixed cost): = {double_load / assumed_build_weight:.2f}")

    print(
        "Expected Score (calculated connector cost + assumed epoxy cost): = {:.2f}".format(
            double_load / total_build_weight
        )
    )


def compare_with_materials(bridge_params):

    raise NotImplementedError("This function is more deprecated than a nokia 3310.")

    """Compare the same bridge design with different materials"""

    print("\n" + "=" * 67)

    print("MATERIAL COMPARISON")

    print("=" * 67 + "\n")

    materials = [
        ("Steel", Steel()),
        ("Aluminum", Aluminum()),
        ("Ochroma Wood", OchromaWood()),
        ("Conservative Balsa", ConservativeBalsaWood()),
        ("High Density Balsa", HighDensityBalsaWood()),
        ("Low Density Balsa", LowDensityBalsaWood()),
    ]

    print(f"{'Material':<25} {'Critical Load':>15} {'Weight':>12} {'Load/Weight':>15}")

    print("-" * 67)

    for mat_name, material in materials:

        test_bridge = OnePanelPratt2D(**bridge_params, material=material)

        failure_dict = test_bridge.get_failure_mode_dict()

        critical_load = min([v for v in failure_dict.values() if v is not None])

        volume = test_bridge.get_total_volume()

        mass = volume * material.density

        material_weight = mass * 9.81

        weight = material_weight + connector_cost

        load_to_weight = critical_load / weight if weight > 0 else float("inf")

        print(
            f"{mat_name:<25} {critical_load:>12.2f} N  {weight:>10.4f} N  {load_to_weight:>15.2f}"
        )


def main():
    """Main function - Run complete bridge analysis"""

    print("\n" + "=" * 67)

    print("MANUAL BRIDGE TESTER")

    print("Design and Analyze Your Custom Bridge")

    print(f"Display Units: {DISPLAY_UNITS.upper()}")

    print("=" * 67)

    bridge, material = create_custom_bridge()

    print_bridge_summary(bridge, material, unit_system=DISPLAY_UNITS)

    make_and_report(
        bridge,
        density=material.density,
        special_message="STANDARD FAILURE ANALYSIS",
        unit_system=DISPLAY_UNITS,
    )

    print_detailed_failure_analysis(bridge, unit_system=DISPLAY_UNITS)

    visualize_bridge(bridge, save_filename="custom_bridge_design.png")

    print_parts_list(bridge, material)

    """
    print("\n" + "="*67)
    print("  ANALYSIS COMPLETE")
    print("="*67 + "\n")

    print("To test a different design:")
    print("  1. Edit the 'create_custom_bridge()' function in this file")
    print("  2. Run: python manual_bridge_tester.py")
    print()"""


if __name__ == "__main__":

    main()
