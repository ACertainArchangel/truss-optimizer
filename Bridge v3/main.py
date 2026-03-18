"""An example of using this very simple and really cool truss optimizer, if I do say so myself."""

import json
import math

from truss_optimizer import BridgeOptimizer, PrattTruss, materials
from truss_optimizer.utils.units import inches_to_meters, kgf_to_newtons

# Create a Pratt truss bridge (from Bridge v2 competition design)
bridge = PrattTruss(
    angle=math.radians(32.78),
    height=inches_to_meters(5.16),
    span=inches_to_meters(18.5),
    material=materials.BalsaWood(),
    incline_thickness=inches_to_meters(0.5),
    incline_depth=inches_to_meters(8 / 16),
    diagonal_thickness=inches_to_meters(0.15934),
    diagonal_depth=inches_to_meters(2 / 16),
    mid_vert_thickness=inches_to_meters(0.500000),
    mid_vert_depth=inches_to_meters(8 / 16),
    side_vert_thickness=inches_to_meters(0.14966),
    side_vert_depth=inches_to_meters(3 / 16),
    top_thickness=inches_to_meters(0.500000),
    top_depth=inches_to_meters(8 / 16),
    bottom_thickness=inches_to_meters(0.48385),
    bottom_depth=inches_to_meters(5 / 16),
)

thickness_bounds = (inches_to_meters(0.1), inches_to_meters(0.5))  # 2.54mm - 12.7mm
depth_bounds = (inches_to_meters(0.1), inches_to_meters(0.5))  # 2.54mm - 12.7mm

constraints = {
    "incline_thickness": thickness_bounds,
    "diagonal_thickness": thickness_bounds,
    "mid_vert_thickness": thickness_bounds,
    "side_vert_thickness": thickness_bounds,
    "top_thickness": thickness_bounds,
    "bottom_thickness": thickness_bounds,
    "incline_depth": depth_bounds,
    "diagonal_depth": depth_bounds,
    "mid_vert_depth": depth_bounds,
    "side_vert_depth": depth_bounds,
    "top_depth": depth_bounds,
    "bottom_depth": depth_bounds,
    "angle": (math.radians(20), math.radians(70)),
    "height": (inches_to_meters(3), inches_to_meters(6)),
}

# Optimize it
optimizer = BridgeOptimizer(
    bridge,
    objective="load_to_weight",
    constraints=constraints,
    fixed_cost=0.12,  # Additional weight (N) for fasteners, glue, etc.
)
result = optimizer.optimize(iterations=1000)

print("ORIGINAL:")
print(
    f"Weight: {bridge.weight+optimizer.fixed_cost:.2f} N"
)  # ADDING fixed cost here but not to the result because v3 is garbage. Still works but just confusing.
print(f"Critical Load: {bridge.critical_load:.2f} N")
print(
    f"Load/Weight: {(bridge.critical_load / (bridge.weight+optimizer.fixed_cost)):.1f}"
)  # Same here

print("\nOPTIMIZED:")
print(
    f"Weight: {result.weight:.2f} N"
)  # Because fixed cost is in the optimizer result. Trash design and this is why we have v4.
print(f"Critical Load: {result.critical_load:.2f} N")
print(
    f"Load/Weight: {(result.critical_load / (result.weight)):.1f}"
)  # Because fixed cost is in the optimizer result. Trash design and this is why we have v4.

print("Parameters:\n", json.dumps(result.params, indent=4))

if True:  # Reqs matplotlib
    result.visualize(show=True)
