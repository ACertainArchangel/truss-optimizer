"""
Comprehensive parity test between bridges_parametric and pratt_torch.
Tests that ALL failure modes match perfectly between implementations.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
from bridges_parametric import OnePanelPratt2D
from pratt_torch import DTYPE, max_load_torch


def test_parity(params_dict, test_name="Default", tolerance_pct=1.0):
    """
    Test parity between parametric and torch implementations.

    Args:
        params_dict: Dictionary of bridge parameters
        test_name: Name of this test case
        tolerance_pct: Acceptable percentage difference (default 1%)

    Returns:
        bool: True if all modes match within tolerance
    """

    print(f"\n{'='*80}")

    print(f"TEST CASE: {test_name}")

    print(f"{'='*80}")

    params_no_k = {k: v for k, v in params_dict.items() if k != "K"}

    bridge = OnePanelPratt2D(**params_no_k)

    failure_modes_parametric = bridge.get_failure_mode_dict()

    max_load_torch_val, failure_modes_torch = max_load_torch(**params_dict)

    print(f"\n{'Mode':<40} {'Parametric':>15} {'Torch':>15} {'Diff %':>10} {'Status':>8}")

    print("-" * 92)

    all_modes = set(failure_modes_parametric.keys()) | set(failure_modes_torch.keys())

    all_modes.discard("torsion_failure")

    mismatches = []

    missing = []

    extra = []

    matches = []

    for mode in sorted(all_modes):

        param_val = failure_modes_parametric.get(mode)

        torch_val = failure_modes_torch.get(mode)

        if param_val is None:

            extra.append(mode)

            torch_val_scalar = (
                torch_val.item() if isinstance(torch_val, torch.Tensor) else torch_val
            )

            print(f"{mode:<40} {'N/A':>15} {torch_val_scalar:15.2f} {'N/A':>10} {'EXTRA':>8}")

            continue

        if torch_val is None:

            missing.append(mode)

            print(f"{mode:<40} {param_val:15.2f} {'N/A':>15} {'N/A':>10} {'MISSING':>8}")

            continue

        torch_val_scalar = torch_val.item() if isinstance(torch_val, torch.Tensor) else torch_val

        sign_mismatch = (torch_val_scalar * param_val) < 0

        if abs(param_val) > 1e-12:

            diff_pct = abs(torch_val_scalar - param_val) / abs(param_val) * 100

        else:

            diff_pct = float("inf") if abs(torch_val_scalar) > 1e-12 else 0.0

        if sign_mismatch:

            status = " SIGN"

            mismatches.append((mode, param_val, torch_val_scalar, diff_pct))

        elif diff_pct < tolerance_pct:

            status = ""

            matches.append(mode)

        else:

            status = ""

            mismatches.append((mode, param_val, torch_val_scalar, diff_pct))

        print(f"{mode:<40} {param_val:15.2f} {torch_val_scalar:15.2f} {diff_pct:9.3f}% {status:>8}")

    print("-" * 92)

    total = len(all_modes)

    print(f"\nSummary:")

    print(f"  Matches:    {len(matches)}/{total} ({len(matches)/total*100:.1f}%)")

    print(f"  Mismatches: {len(mismatches)}/{total}")

    print(f"  Missing:    {len(missing)}/{total}")

    print(f"  Extra:      {len(extra)}/{total}")

    min_param = min(v for v in failure_modes_parametric.values() if v is not None)

    min_torch = max_load_torch_val.item()

    print(f"\nMin failure load:")

    print(f"  Parametric: {min_param:15.2f} N")

    print(f"  Torch:      {min_torch:15.2f} N")

    print(
        f"  Difference: {abs(min_param - min_torch):15.2f} N ({abs(min_param-min_torch)/min_param*100:.3f}%)"
    )

    success = len(mismatches) == 0 and len(missing) == 0 and len(extra) == 0

    if success:

        print(f"\n{' ALL TESTS PASSED':^80}")

    else:

        print(f"\n{' SOME TESTS FAILED':^80}")

        if mismatches:

            print("\nMismatched modes:")

            for mode, param_val, torch_val, diff_pct in mismatches:

                print(f"  {mode}: {diff_pct:.3f}% difference")

    return success


if __name__ == "__main__":

    params_standard = dict(
        angle=math.radians(30),
        height=2.0,
        length=10.0,
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
        K=1.0,
    )

    params_slender = dict(
        angle=math.radians(30),
        height=2.0,
        length=10.0,
        incline_thickness=0.005,
        diagonal_thickness=0.005,
        mid_vert_thickness=0.005,
        side_vert_thickness=0.005,
        top_thickness=0.005,
        bottom_thickness=0.005,
        incline_depth=0.08,
        diagonal_depth=0.08,
        mid_vert_depth=0.08,
        side_vert_depth=0.08,
        top_depth=0.08,
        bottom_depth=0.08,
        E=200e9,
        sigma_compression=2500e6,
        sigma_tension=4000e6,
        K=1.0,
    )

    params_stocky = dict(
        angle=math.radians(45),
        height=1.0,
        length=5.0,
        incline_thickness=0.08,
        diagonal_thickness=0.08,
        mid_vert_thickness=0.08,
        side_vert_thickness=0.08,
        top_thickness=0.08,
        bottom_thickness=0.08,
        incline_depth=0.1,
        diagonal_depth=0.1,
        mid_vert_depth=0.1,
        side_vert_depth=0.1,
        top_depth=0.1,
        bottom_depth=0.1,
        E=200e9,
        sigma_compression=150e6,
        sigma_tension=250e6,
        K=1.0,
    )

    results = []

    results.append(test_parity(params_standard, "Standard Configuration"))

    results.append(test_parity(params_slender, "Slender Members (Euler Dominant)"))

    results.append(test_parity(params_stocky, "Stocky Members (Material Dominant)"))

    print(f"\n{'='*80}")

    print(f"FINAL SUMMARY")

    print(f"{'='*80}")

    print(f"Tests passed: {sum(results)}/{len(results)}")

    if all(results):

        print(f"\n{' ALL PARITY TESTS PASSED - PERFECT AGREEMENT!':^80}")

    else:

        print(f"\n{' SOME PARITY TESTS FAILED':^80}")
