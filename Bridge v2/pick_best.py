"""Evaluate all optimized bridges with a specified material and pick the best one."""

import json
import sys
from pathlib import Path

from bridges_parametric import OnePanelPratt2D
from main import fixed_cost
from materials import (
    ConservativeBalsaWood,
    HighDensityBalsaWood,
    LowDensityBalsaWood,
    OchromaWithEpoxy,
    OchromaWood,
)


def evaluate_all_bridges(material, trials_dir="trials"):
    """Load all bridge JSON files, evaluate with specified material, and find the best."""

    trials_path = Path(trials_dir)

    json_files = sorted(trials_path.glob("final_pratt_bridge*.json"))

    if not json_files:

        raise FileNotFoundError(f"No bridge JSON files found in {trials_dir}/")

    print(f"Evaluating {len(json_files)} bridges with {material.__class__.__name__} material...")

    print("=" * 80)

    results = []

    for json_file in json_files:

        with open(json_file, "r") as f:

            params = json.load(f)

        bridge = OnePanelPratt2D(**params)

        failure_dict = bridge.get_failure_mode_dict()

        critical_load = min([v for v in failure_dict.values() if v is not None])

        governing_mode = min(
            [(k, v) for k, v in failure_dict.items() if v is not None], key=lambda x: x[1]
        )

        volume = bridge.get_total_volume()

        mass = volume * material.density

        material_weight = mass * 9.81

        weight = material_weight + fixed_cost

        load_weight_ratio = critical_load / weight if weight > 0 else 0

        results.append(
            {
                "file": json_file.name,
                "seed": int(json_file.stem.replace("final_pratt_bridge", "")),
                "critical_load": critical_load,
                "governing_mode": governing_mode[0],
                "weight": weight,
                "load_weight_ratio": load_weight_ratio,
                "params": params,
            }
        )

    results.sort(key=lambda x: x["load_weight_ratio"], reverse=True)

    return results


def print_top_results(results, n=10):
    """Print the top N results."""

    print(f"\nTOP {n} BRIDGES (by load-to-weight ratio):")

    print("-" * 80)

    print(
        f"{'Rank':<6} {'Seed':<6} {'Load (N)':<12} {'Weight (N)':<12} {'Ratio':<10} {'Governing Mode'}"
    )

    print("-" * 80)

    for i, result in enumerate(results[:n], 1):

        print(
            f"{i:<6} {result['seed']:<6} {result['critical_load']:>10.2f}  "
            f"{result['weight']:>10.4f}  {result['load_weight_ratio']:>8.1f}  "
            f"{result['governing_mode']}"
        )

    print("-" * 80)


def save_best_bridge(result, output_file="best_bridge.json"):
    """Save the best bridge parameters to a file."""

    with open(output_file, "w") as f:

        json.dump(result["params"], f, indent=2)

    print(f"\nBest bridge saved to '{output_file}'")


def print_detailed_parameters(params, unit="in"):
    """Print parameters in organized format like post_process_parameters_perfectly."""

    import math

    if unit == "in":

        conv = 39.3701

        unit_name = "inches"

    elif unit == "cm":

        conv = 100.0

        unit_name = "centimeters"

    else:

        conv = 1.0

        unit_name = "meters"

    print(f"\nUnits: distances in [{unit}], angle in [degrees]")

    geometric_params = [
        ("angle", "radians_to_degrees"),
        ("height", "length"),
        ("length", "length"),
        ("incline_thickness", "length"),
        ("incline_depth", "length"),
        ("diagonal_thickness", "length"),
        ("diagonal_depth", "length"),
        ("mid_vert_thickness", "length"),
        ("mid_vert_depth", "length"),
        ("side_vert_thickness", "length"),
        ("side_vert_depth", "length"),
        ("top_thickness", "length"),
        ("top_depth", "length"),
        ("bottom_thickness", "length"),
        ("bottom_depth", "length"),
    ]

    material_params = ["E", "sigma_compression", "sigma_tension"]

    print("\nGEOMETRIC PARAMETERS:")

    print(f"{'Parameter':<25} {'Value':>15}")

    print("-" * 42)

    for param, conv_type in geometric_params:

        if param in params:

            if conv_type == "radians_to_degrees":

                value = params[param] * 180.0 / math.pi

                print(f"{param:<25} {value:15.6f}")

            else:

                value = params[param] * conv

                print(f"{param:<25} {value:15.6f}")

    print("\nMATERIAL PARAMETERS:")

    print(f"{'Parameter':<25} {'Value':>15}")

    print("-" * 42)

    for param in material_params:

        if param in params:

            value = params[param]

            if isinstance(value, (int, float)):

                print(f"{param:<25} {value:15.2e}")

            else:

                print(f"{param:<25} {str(value):>15}")

    print(f"material {material_name:>35}")


def main(material_name="OchromaWithEpoxy"):
    """Main function to evaluate and pick the best bridge."""

    materials = {
        "OchromaWood": OchromaWood,
        "ConservativeBalsaWood": ConservativeBalsaWood,
        "HighDensityBalsaWood": HighDensityBalsaWood,
        "LowDensityBalsaWood": LowDensityBalsaWood,
        "OchromaWithEpoxy": OchromaWithEpoxy,
    }

    if material_name not in materials:

        print(f"Error: Unknown material '{material_name}'")

        print(f"Available materials: {', '.join(materials.keys())}")

        return None

    material = materials[material_name]()

    print("=" * 80)

    print(f"BEST BRIDGE SELECTION - {material.__class__.__name__.upper()}")

    print("=" * 80)

    results = evaluate_all_bridges(material)

    print(f"\nEvaluated {len(results)} bridges")

    print(
        f"Load-to-weight ratio range: {results[-1]['load_weight_ratio']:.1f} to {results[0]['load_weight_ratio']:.1f}"
    )

    print_top_results(results, n=10)

    best = results[0]

    print(f"\nBEST BRIDGE (Seed {best['seed']}):")

    print("=" * 80)

    print(f"  Critical Load:    {best['critical_load']:>10.2f} N")

    print(f"  Weight:           {best['weight']:>10.4f} N")

    print(f"  Load/Weight:      {best['load_weight_ratio']:>10.1f}")

    print(f"Governing Mode:   {best['governing_mode']}")

    print(f"Source File:      {best['file']}")

    print_detailed_parameters(best["params"], unit="in")

    print("\n" + "=" * 80)

    save_best_bridge(best)

    return results


if __name__ == "__main__":

    material_name = sys.argv[1] if len(sys.argv) > 1 else "OchromaWithEpoxy"

    main(material_name)
