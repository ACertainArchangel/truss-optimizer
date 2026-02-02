"""An example of using this very simple and really cool truss optimizer, if I do say so myself."""

import json
import math
from truss_optimizer import BridgeOptimizer, PrattTruss, materials
from truss_optimizer.utils.units import inches_to_meters

# Create a Pratt truss bridge
bridge = PrattTruss(
    span=0.47,        # meters (0.47 m = 18 inches ish)
    height=0.15,      # meters (0.15 m = 6 inches ish, a good starting point)
    material=materials.BalsaWood()
)

# Competition constraints: max member dimension is 0.5 inches (12.7mm)
# Adjust these bounds based on your competition rules or material availability
thickness_bounds = (inches_to_meters(0.1), inches_to_meters(0.5))  # 2.54mm - 12.7mm
depth_bounds = (inches_to_meters(0.1), inches_to_meters(0.5))      # 2.54mm - 12.7mm

constraints = {
    # Thickness constraints for each member type
    'incline_thickness': thickness_bounds,
    'diagonal_thickness': thickness_bounds,
    'mid_vert_thickness': thickness_bounds,
    'side_vert_thickness': thickness_bounds,
    'top_thickness': thickness_bounds,
    'bottom_thickness': thickness_bounds,
    # Depth constraints for each member type
    'incline_depth': depth_bounds,
    'diagonal_depth': depth_bounds,
    'mid_vert_depth': depth_bounds,
    'side_vert_depth': depth_bounds,
    'top_depth': depth_bounds,
    'bottom_depth': depth_bounds,
    # Geometric constraints
    'angle': (math.radians(20), math.radians(70)),
    'height': (inches_to_meters(3), inches_to_meters(6)),  # Competition max height was 6 inches and 3 is rediculously low
}

# Optimize it
optimizer = BridgeOptimizer(
    bridge, 
    objective='load_to_weight',
    constraints=constraints,
    fixed_cost=0.2  # Additional weight (kg) for fasteners, glue, etc.
)
result = optimizer.optimize(iterations=5000)

print("ORIGINAL:")
print(f"Weight: {bridge.weight:.2f} kg")
print(f"Critical Load: {bridge.critical_load:.2f} N")
print(f"Load/Weight: {bridge.load_to_weight:.1f}")

print("\nOPTIMIZED:")
print(f"Weight: {result.weight:.2f} kg")
print(f"Critical Load: {result.critical_load:.2f} N")
print(f"Load/Weight: {result.load_to_weight:.1f}")

print("Parameters:\n", json.dumps(result.params, indent=4))

if True:
    # Visualize the optimized truss (requires matplotlib)
    result.visualize(show=True)