"""Post-process optimization trials: compute stats and create final_plans.json."""

import json
import math
import os
from pathlib import Path

import numpy as np


def load_trial_data(trials_dir="trials"):
    """Load all trial JSON files from the specified directory"""

    trials_path = Path(trials_dir)

    json_files = sorted(trials_path.glob("final_pratt_bridge*.json"))

    if not json_files:

        raise FileNotFoundError(f"No trial JSON files found in {trials_dir}/")

    print(f"Loading {len(json_files)} trial files...")

    trials = []

    for json_file in json_files:

        with open(json_file, "r") as f:

            trial_data = json.load(f)

            trials.append(trial_data)

    return trials


def compute_statistics(trials):
    """Compute means and standard deviations across all trials"""

    param_keys = list(trials[0].keys())

    param_values = {key: [] for key in param_keys}

    for trial in trials:

        for key in param_keys:

            param_values[key].append(trial[key])

    means = {}

    std_devs = {}

    for key in param_keys:

        values = np.array(param_values[key])

        means[key] = float(np.mean(values))

        std_devs[key] = float(np.std(values, ddof=1))

    return means, std_devs


def convert_units(data, target_unit):
    """Convert data from meters/radians to target units"""

    converted = {}

    m_to_cm = 100.0

    m_to_in = 39.3701

    m_to_ft = 3.28084

    rad_to_deg = 180.0 / math.pi

    if target_unit == "cm":

        length_factor = m_to_cm

        unit_name = "centimeters"

    elif target_unit == "in":

        length_factor = m_to_in

        unit_name = "inches"

    elif target_unit == "ft":

        length_factor = m_to_ft

        unit_name = "feet"

    else:

        raise ValueError(f"Unknown unit: {target_unit}")

    distance_params = [
        "height",
        "length",
        "incline_thickness",
        "incline_depth",
        "diagonal_thickness",
        "diagonal_depth",
        "mid_vert_thickness",
        "mid_vert_depth",
        "side_vert_thickness",
        "side_vert_depth",
        "top_thickness",
        "top_depth",
        "bottom_thickness",
        "bottom_depth",
    ]

    for key, value in data.items():

        if key == "angle":

            converted[key] = value * rad_to_deg

        elif key in distance_params:

            converted[key] = value * length_factor

        else:

            converted[key] = value

    return converted, unit_name


def save_final_plans(means, std_devs, num_trials, output_file="final_plans.json"):
    """Save computed statistics to JSON file with nice formatting"""

    final_plans = {
        "means": means,
        "standard_deviations": std_devs,
        "metadata": {
            "num_trials": num_trials,
            "description": "Aggregated statistics from optimization trials",
            "units": {
                "angle": "radians",
                "height": "meters",
                "length": "meters",
                "thicknesses": "meters",
                "depths": "meters",
                "E": "Pascals",
                "sigma_compression": "Pascals",
                "sigma_tension": "Pascals",
            },
        },
    }

    print(f"Saving results to {output_file}...")

    with open(output_file, "w") as f:

        json.dump(final_plans, f, indent=2, sort_keys=False)

    print(f"Successfully saved to {output_file}")


def save_converted_plans(means, std_devs, num_trials, target_unit):
    """Save converted units version of final plans"""

    means_converted, unit_name = convert_units(means, target_unit)

    std_devs_converted, _ = convert_units(std_devs, target_unit)

    angle_unit = "degrees"

    final_plans = {
        "means": means_converted,
        "standard_deviations": std_devs_converted,
        "metadata": {
            "num_trials": num_trials,
            "description": f"Aggregated statistics from optimization trials (converted to {unit_name})",
            "units": {
                "angle": angle_unit,
                "height": unit_name,
                "length": unit_name,
                "thicknesses": unit_name,
                "depths": unit_name,
                "E": "Pascals",
                "sigma_compression": "Pascals",
                "sigma_tension": "Pascals",
            },
        },
    }

    output_file = f"final_plans_{target_unit.upper()}.json"

    print(f"Saving converted results to {output_file}...")

    with open(output_file, "w") as f:

        json.dump(final_plans, f, indent=2, sort_keys=False)

    print(f"Successfully saved to {output_file}")


def print_summary(means, std_devs, display_unit="m"):
    """Print a nice summary of the results in specified units"""

    print("\n" + "=" * 80)

    print("OPTIMIZATION RESULTS SUMMARY")

    print("=" * 80)

    if display_unit in ["cm", "in", "ft"]:

        means_display, unit_name = convert_units(means, display_unit)

        std_devs_display, _ = convert_units(std_devs, display_unit)

        angle_unit = "degrees"

        length_unit = display_unit

    else:

        means_display = means

        std_devs_display = std_devs

        unit_name = "meters"

        angle_unit = "radians"

        length_unit = "m"

    print(f"\nUnits: distances in [{length_unit}], angle in [{angle_unit}]")

    geometric_params = [
        "angle",
        "height",
        "length",
        "incline_thickness",
        "incline_depth",
        "diagonal_thickness",
        "diagonal_depth",
        "mid_vert_thickness",
        "mid_vert_depth",
        "side_vert_thickness",
        "side_vert_depth",
        "top_thickness",
        "top_depth",
        "bottom_thickness",
        "bottom_depth",
    ]

    material_params = ["E", "sigma_compression", "sigma_tension"]

    print("\nGEOMETRIC PARAMETERS:")

    print(f"{'Parameter':<25} {'Mean':>15} {'Std Dev':>15} {'CV %':>10}")

    print("-" * 68)

    for param in geometric_params:

        if param in means_display:

            mean = means_display[param]

            std = std_devs_display[param]

            cv = (std_devs[param] / means[param] * 100) if means[param] != 0 else 0

            print(f"{param:<25} {mean:15.6f} {std:15.6f} {cv:9.2f}%")

    print("\nMATERIAL PARAMETERS:")

    print(f"{'Parameter':<25} {'Mean':>15} {'Std Dev':>15} {'CV %':>10}")

    print("-" * 68)

    for param in material_params:

        if param in means_display:

            mean = means_display[param]

            std = std_devs_display[param]

            cv = (std / mean * 100) if mean != 0 else 0

            print(f"{param:<25} {mean:15.2e} {std:15.2e} {cv:9.2f}%")

    print("\n" + "=" * 80)


def main(display_unit="m"):
    """Run post-processing pipeline for all trial data."""

    print("=" * 80)

    print("BRIDGE OPTIMIZATION POST-PROCESSING")

    print("=" * 80)

    trials = load_trial_data("trials")

    num_trials = len(trials)

    means, std_devs = compute_statistics(trials)

    save_final_plans(means, std_devs, num_trials, "final_plans.json")

    save_converted_plans(means, std_devs, num_trials, "cm")

    save_converted_plans(means, std_devs, num_trials, "in")

    print_summary(means, std_devs, display_unit)

    print("\nPost-processing complete!")

    print(f"Processed {num_trials} trials")

    print(f"Results saved to:")

    print(f"   - final_plans.json (SI units: meters, radians)")

    print(f"   - final_plans_CM.json (centimeters, degrees)")

    print(f"   - final_plans_IN.json (inches, degrees)")


if __name__ == "__main__":

    import sys

    display_unit = sys.argv[1] if len(sys.argv) > 1 else "m"

    if display_unit not in ["m", "cm", "in", "ft"]:

        print(f"Warning: Unknown unit '{display_unit}', using 'm' (meters)")

        display_unit = "m"

    main(display_unit)
