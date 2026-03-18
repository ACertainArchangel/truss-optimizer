"""Optimize manual bridge tester config: height, angle, and thicknesses."""

import json
import math
import re
import sys
from dataclasses import asdict

from main import epoxy_cost
from materials import OchromaWithEpoxy, OchromaWood
from pratt_optimizer import BridgeDesignParams, BridgeOptimizer
from utils import inches_to_meters

sys.path.insert(0, ".")


def extract_params_from_manual_tester():
    """Extract current parameters from manual_bridge_tester.py using regex"""

    with open("manual_bridge_tester.py", "r") as f:

        content = f.read()

    params = {}

    def _parse_inches_token(token: str) -> float:

        token = token.strip()

        if "/" in token:

            parts = [p.strip() for p in token.split("/") if p.strip()]

            if len(parts) == 2:

                try:

                    num = float(parts[0])

                    den = float(parts[1])

                    return num / den

                except Exception as e:

                    raise ValueError(f"Unable to parse fraction token '{token}': {e}")

            else:

                raise ValueError(f"Unsupported fraction format: '{token}'")

        try:

            return float(token)

        except Exception as e:

            raise ValueError(f"Unable to parse numeric token '{token}': {e}")

    angle_match = re.search(r"angle\s*=\s*math\.radians\(\s*([0-9.]+)\s*\)", content)

    if angle_match:

        params["angle"] = math.radians(float(angle_match.group(1)))

    else:

        raise ValueError("Could not extract 'angle' from manual_bridge_tester.py")

    height_match = re.search(r"height\s*=\s*inches_to_meters\(\s*([0-9./ ]+)\s*\)", content)

    if height_match:

        token = height_match.group(1)

        params["height"] = inches_to_meters(_parse_inches_token(token))

    else:

        raise ValueError("Could not extract 'height' from manual_bridge_tester.py")

    length_match = re.search(r"length\s*=\s*inches_to_meters\(\s*([0-9./ ]+)\s*\)", content)

    if length_match:

        token = length_match.group(1)

        params["length"] = inches_to_meters(_parse_inches_token(token))

    else:

        raise ValueError("Could not extract 'length' from manual_bridge_tester.py")

    param_names = [
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

    for param_name in param_names:

        match = re.search(rf"{param_name}\s*=\s*inches_to_meters\(\s*([0-9./ ]+)\s*\)", content)

        if match:

            token = match.group(1)

            params[param_name] = inches_to_meters(_parse_inches_token(token))

        else:

            raise ValueError(f"Could not extract '{param_name}' from manual_bridge_tester.py")

    print("Successfully extracted all parameters")

    return params


def optimize_manual_bridge():
    """Optimize the bridge from manual_bridge_tester.py"""

    print("=" * 80)

    print("OPTIMIZING MANUAL BRIDGE TESTER CONFIGURATION")

    print("=" * 80)

    material = OchromaWithEpoxy()

    print("\nExtracting parameters from manual_bridge_tester.py...")

    extracted_params = extract_params_from_manual_tester()

    initial_params = BridgeDesignParams(
        angle=extracted_params["angle"],
        height=extracted_params["height"],
        length=extracted_params["length"],
        incline_thickness=extracted_params["incline_thickness"],
        diagonal_thickness=extracted_params["diagonal_thickness"],
        mid_vert_thickness=extracted_params["mid_vert_thickness"],
        side_vert_thickness=extracted_params["side_vert_thickness"],
        top_thickness=extracted_params["top_thickness"],
        bottom_thickness=extracted_params["bottom_thickness"],
        incline_depth=extracted_params["incline_depth"],
        diagonal_depth=extracted_params["diagonal_depth"],
        mid_vert_depth=extracted_params["mid_vert_depth"],
        side_vert_depth=extracted_params["side_vert_depth"],
        top_depth=extracted_params["top_depth"],
        bottom_depth=extracted_params["bottom_depth"],
        E=material.E,
        sigma_compression=material.sigma_compression,
        sigma_tension=material.sigma_tension,
        density=material.density,
        material=material,
    )

    fixed_params = {
        "E": True,
        "sigma_compression": True,
        "sigma_tension": True,
        "density": True,
        "length": True,
        "incline_depth": True,
        "diagonal_depth": True,
        "mid_vert_depth": True,
        "side_vert_depth": True,
        "top_depth": True,
        "bottom_depth": True,
        "angle": False,
        "height": False,
        "incline_thickness": False,
        "diagonal_thickness": False,
        "mid_vert_thickness": False,
        "side_vert_thickness": False,
        "top_thickness": False,
        "bottom_thickness": False,
    }

    min_thickness = inches_to_meters(0.05)

    max_thickness = inches_to_meters(0.5)

    param_bounds = {
        "length": (extracted_params["length"], extracted_params["length"]),
        "angle": (math.radians(30), math.radians(85)),
        "height": (inches_to_meters(4.0), inches_to_meters(10.0)),
        "incline_thickness": (min_thickness, max_thickness),
        "diagonal_thickness": (min_thickness, max_thickness),
        "mid_vert_thickness": (min_thickness, max_thickness),
        "side_vert_thickness": (min_thickness, max_thickness),
        "top_thickness": (min_thickness, max_thickness),
        "bottom_thickness": (min_thickness, max_thickness),
        "incline_depth": (extracted_params["incline_depth"], extracted_params["incline_depth"]),
        "diagonal_depth": (extracted_params["diagonal_depth"], extracted_params["diagonal_depth"]),
        "mid_vert_depth": (extracted_params["mid_vert_depth"], extracted_params["mid_vert_depth"]),
        "side_vert_depth": (
            extracted_params["side_vert_depth"],
            extracted_params["side_vert_depth"],
        ),
        "top_depth": (extracted_params["top_depth"], extracted_params["top_depth"]),
        "bottom_depth": (extracted_params["bottom_depth"], extracted_params["bottom_depth"]),
    }

    print("\nInitial Configuration:")

    print(f"Angle:  {math.degrees(initial_params.angle):.2f}°")

    print(f"Height: {initial_params.height/0.0254:.2f} in")

    print(f"Length: {initial_params.length/0.0254:.2f} in (FIXED)")

    print(f"\nThicknesses (trainable):")

    print(f"Incline:  {initial_params.incline_thickness/0.0254:.3f} in")

    print(f"Diagonal: {initial_params.diagonal_thickness/0.0254:.3f} in")

    print(f"Mid Vert: {initial_params.mid_vert_thickness/0.0254:.3f} in")

    print(f"Side Vert: {initial_params.side_vert_thickness/0.0254:.3f} in")

    print(f"Top:      {initial_params.top_thickness/0.0254:.3f} in")

    print(f"Bottom:   {initial_params.bottom_thickness/0.0254:.3f} in")

    print(f"\nDepths (FIXED):")

    print(f"Incline:  {initial_params.incline_depth/0.0254:.3f} in")

    print(f"Diagonal: {initial_params.diagonal_depth/0.0254:.3f} in")

    print(f"Mid Vert: {initial_params.mid_vert_depth/0.0254:.3f} in")

    print(f"Side Vert: {initial_params.side_vert_depth/0.0254:.3f} in")

    print(f"Top:      {initial_params.top_depth/0.0254:.3f} in")

    print(f"Bottom:   {initial_params.bottom_depth/0.0254:.3f} in")

    optimizer = BridgeOptimizer(
        initial_params=initial_params,
        fixed_params=fixed_params,
        param_bounds=param_bounds,
        learning_rate=0.001,
        material=material,
        fixed_cost=epoxy_cost,
        verbose=False,
    )

    print(f"\nStarting optimization for 7000 iterations...")

    print("Optimizing: angle, height, and all 6 thicknesses")

    print("=" * 80 + "\n")

    try:

        init_params, final_params = optimizer.train(
            num_iterations=7000, verbose=True, log_interval=100
        )

        print(f"\nDEBUG: best_params angle: {math.degrees(final_params['angle']):.6f}°")

        print(
            f"DEBUG: best_params incline_thick: {final_params['incline_thickness']/0.0254:.6f} in"
        )

        print(f"DEBUG: best_iteration: {optimizer.best_iteration}")

        print(f"DEBUG: best_objective_value: {optimizer.best_objective_value}")

    except KeyboardInterrupt:

        print("\n\nOptimization interrupted by user!")

        init_params = optimizer.get_param_values()

        final_params = init_params

    print("\n" + "=" * 80)

    print("OPTIMIZATION COMPLETE")

    print("=" * 80)

    print("\nBEST Configuration (from iteration with highest load/weight):")

    print(f"Angle:  {math.degrees(final_params['angle']):.2f}°")

    print(f"Height: {final_params['height']/0.0254:.2f} in")

    print(f"\nOptimized Thicknesses:")

    print(f"Incline:  {final_params['incline_thickness']/0.0254:.3f} in")

    print(f"Diagonal: {final_params['diagonal_thickness']/0.0254:.3f} in")

    print(f"Mid Vert: {final_params['mid_vert_thickness']/0.0254:.3f} in")

    print(f"Side Vert: {final_params['side_vert_thickness']/0.0254:.3f} in")

    print(f"Top:      {final_params['top_thickness']/0.0254:.3f} in")

    print(f"Bottom:   {final_params['bottom_thickness']/0.0254:.3f} in")

    with open("optimized_manual_bridge.json", "w") as f:

        json.dump(final_params, f, indent=2)

    print(f"\nSaved optimized parameters to 'optimized_manual_bridge.json'")

    return final_params


def update_manual_bridge_tester(params):
    """Update manual_bridge_tester.py with optimized parameters"""

    print("\n" + "=" * 80)

    print("UPDATING manual_bridge_tester.py")

    print("=" * 80)

    with open("manual_bridge_tester.py", "r") as f:

        content = f.read()

    import re

    replacements = {
        r"angle = math\.radians\([^)]+\)": f"angle = math.radians({math.degrees(params['angle']):.6f})",
        r"height = inches_to_meters\([^)]+\)": f"height = inches_to_meters({params['height']/0.0254:.6f})",
        r"incline_thickness = inches_to_meters\([^)]+\)": f"incline_thickness = inches_to_meters({params['incline_thickness']/0.0254:.6f})",
        r"diagonal_thickness = inches_to_meters\([^)]+\)": f"diagonal_thickness = inches_to_meters({params['diagonal_thickness']/0.0254:.6f})",
        r"mid_vert_thickness = inches_to_meters\([^)]+\)": f"mid_vert_thickness = inches_to_meters({params['mid_vert_thickness']/0.0254:.6f})",
        r"side_vert_thickness = inches_to_meters\([^)]+\)": f"side_vert_thickness = inches_to_meters({params['side_vert_thickness']/0.0254:.6f})",
        r"top_thickness = inches_to_meters\([^)]+\)": f"top_thickness = inches_to_meters({params['top_thickness']/0.0254:.6f})",
        r"bottom_thickness = inches_to_meters\([^)]+\)": f"bottom_thickness = inches_to_meters({params['bottom_thickness']/0.0254:.6f})",
    }

    for pattern, new in replacements.items():

        matches = re.findall(pattern, content)

        if matches:

            content = re.sub(pattern, new, content)

            param_name = pattern.split("=")[0].replace("\\", "").strip().replace("r", "")

            print(f"Updated: {param_name}")

    with open("manual_bridge_tester.py", "w") as f:

        f.write(content)

    print("\nmanual_bridge_tester.py updated with optimized parameters!")


def run_analysis():
    """Run the analysis on the optimized bridge"""

    print("\n" + "=" * 80)

    print("RUNNING ANALYSIS ON OPTIMIZED BRIDGE")

    print("=" * 80 + "\n")

    import manual_bridge_tester

    manual_bridge_tester.main()


if __name__ == "__main__":

    final_params = optimize_manual_bridge()

    update_manual_bridge_tester(final_params)

    run_analysis()

    print("\n" + "=" * 80)

    print("ALL DONE!")

    print("=" * 80)

    print("\nYour optimized bridge parameters are now in manual_bridge_tester.py")

    print("You can run it again anytime with: python manual_bridge_tester.py")
