"""
Compute total surface area of lateral bracing members for manufacturing
(manual input)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

laterals = [
    (0.18750, 5.00000, 9, "T-lats"),
    (0.18750, 9.42949, 6, "T-mid/s"),
    (0.18750, 9.63358, 6, "B-mid/s"),
    (0.18750, 5.37500, 9, "B-lats"),
    (0.18750, 5.93243, 6, "B-out/s"),
]

print("Lateral Bracing Surface Area Calculation")
print("=" * 67)
print(f"{'Member':<15} {'Width (in)':<12} {'Length (in)':<12} {'Qty':<6} {'Area (in^2)'}")
print("-" * 67)

total_area = 0.0

for width, length, qty, name in laterals:

    area = width * length * qty
    total_area += area

    print(f"{name:<15} {width:<12.5f} {length:<12.5f} {qty:<6} {area:>10.2f}")

print("-" * 67)
print(f"{'TOTAL':<15} {'':<12} {'':<12} {'':<6} {total_area:>10.2f}")
print(f"\nTotal surface area of lateral bracing: {total_area:.2f} in^2")

cubic_inches = total_area * (1 / 16)
cubic_meters = cubic_inches * 0.0000163871

from materials import OchromaWithEpoxy

density = OchromaWithEpoxy().density
density = 720

print(f"Density of Ochroma with Epoxy: {density} kg/m^3")

mass_kg = cubic_meters * density
weight_N = mass_kg * 9.81

print(f"Total weight of lateral bracing: {weight_N:.4f} N")
print(f"Total volume of lateral bracing: {cubic_meters:.10f} m^3")
