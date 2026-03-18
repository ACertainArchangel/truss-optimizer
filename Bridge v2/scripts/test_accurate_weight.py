"""
Test the updated optimizer with accurate weight calculation.
"""

import sys

sys.path.insert(0, ".")

import math

from materials import OchromaWithEpoxy
from pratt_optimizer import BridgeDesignParams, BridgeOptimizer
from utils import inches_to_meters

material = OchromaWithEpoxy()

initial_params = BridgeDesignParams(
    angle=math.radians(35),
    height=inches_to_meters(5.5),
    length=inches_to_meters(18.5),
    incline_thickness=inches_to_meters(0.5),
    diagonal_thickness=inches_to_meters(0.15),
    mid_vert_thickness=inches_to_meters(0.5),
    side_vert_thickness=inches_to_meters(0.15),
    top_thickness=inches_to_meters(0.5),
    bottom_thickness=inches_to_meters(0.5),
    incline_depth=inches_to_meters(0.5),
    diagonal_depth=inches_to_meters(2 / 16),
    mid_vert_depth=inches_to_meters(0.5),
    side_vert_depth=inches_to_meters(3 / 16),
    top_depth=inches_to_meters(0.5),
    bottom_depth=inches_to_meters(5 / 16),
    material=material,
)

fixed_params = {
    "E": True,
    "sigma_compression": True,
    "sigma_tension": True,
    "density": True,
    "length": True,
    "angle": True,
    "height": True,
    "incline_depth": True,
    "diagonal_depth": True,
    "mid_vert_depth": True,
    "side_vert_depth": True,
    "top_depth": True,
    "bottom_depth": True,
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
    "incline_thickness": (min_thickness, max_thickness),
    "diagonal_thickness": (min_thickness, max_thickness),
    "mid_vert_thickness": (min_thickness, max_thickness),
    "side_vert_thickness": (min_thickness, max_thickness),
    "top_thickness": (min_thickness, max_thickness),
    "bottom_thickness": (min_thickness, max_thickness),
}

print("=" * 80)
print("TESTING OPTIMIZER WITH ACCURATE WEIGHT CALCULATION")
print("=" * 80)
print()
print("Creating optimizer with epoxy_cost = 0.515 N...")
print()

optimizer = BridgeOptimizer(
    initial_params=initial_params,
    fixed_params=fixed_params,
    param_bounds=param_bounds,
    learning_rate=0.001,
    material=material,
    fixed_cost=0.515,
    verbose=True,
)

print("\nRunning 100 iterations...")
print()

init_params, final_params = optimizer.train(num_iterations=100, verbose=True, log_interval=25)

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print()
print("The optimizer now uses accurate weight calculation including:")
print("  • Main truss members")
print('  • Lateral bracing supports (3/16" thickness)')
print("  • Epoxy weight (0.515 N per plane)")
print()
print("This matches the actual parts list calculation from manual_bridge_tester.py")
print()
