"""
Debug: Compare volume calculations line by line
"""

import sys

sys.path.insert(0, ".")
import math

import torch
from materials import OchromaWithEpoxy
from utils import inches_to_meters

angle = math.radians(32.78)
height_in = 5.16
len_in = 18.5
incline_thick = 0.500000
diagonal_thick = 0.15934
mid_vert_thick = 0.500000
side_vert_thick = 0.14966
top_thick = 0.500000
bottom_thick = 0.48385
incline_depth = 8 / 16
diagonal_depth = 2 / 16
mid_vert_depth = 8 / 16
side_vert_depth = 3 / 16
top_depth = 8 / 16
bottom_depth = 5 / 16


def pythag(a, b):
    # This was Pythagoras's tag name back in the 50's (the 550's BC) when he was a
    # graffiti artist in the streets of Samos, Greece. He would write "pythag(a, b)"
    # next to his drawings of right triangles, and it was only centuries later that
    # we realised this young punk from Samos was referring to the Pythagorean theorem.
    return math.sqrt(a**2 + b**2)


print("=" * 67)
print("MANUAL CALCULATION (inches)")
print("=" * 67)

incline_length = height_in / math.sin(angle)
phi = math.atan(math.tan(angle) / 2)
diagonal_length = height_in / math.cos(phi)
top_length = len_in - 2 * height_in / math.tan(angle)
bottom_length = len_in
vert_length = height_in - bottom_thick - top_thick

print(f"incline_length:  {incline_length:.8f} in")
print(f"diagonal_length: {diagonal_length:.8f} in")
print(f"top_length:      {top_length:.8f} in")
print(f"bottom_length:   {bottom_length:.8f} in")
print(f"vert_length:     {vert_length:.8f} in")

v_incline = 4 * (incline_thick * incline_length * incline_depth)
v_diagonal = 4 * (diagonal_thick * diagonal_length * diagonal_depth)
v_top = 2 * (top_thick * top_length * top_depth)
v_bottom = 2 * (bottom_thick * bottom_length * bottom_depth)
v_mid_vert = 2 * (mid_vert_thick * vert_length * mid_vert_depth)
v_side_vert = 4 * (side_vert_thick * vert_length * side_vert_depth)

print(f"\nMain member volumes (in^3):")
print(f"incline:   {v_incline:.8f}")
print(f"diagonal:  {v_diagonal:.8f}")
print(f"top:       {v_top:.8f}")
print(f"bottom:    {v_bottom:.8f}")
print(f"mid_vert:  {v_mid_vert:.8f}")
print(f"side_vert: {v_side_vert:.8f}")

main_volume_in = v_incline + v_diagonal + v_top + v_bottom + v_mid_vert + v_side_vert

print(f"  TOTAL:     {main_volume_in:.8f} in^3")

t_lat_length = 6 - 2 * top_depth
t_mid_length = pythag(height_in / math.tan(angle), 6 - 2 * top_depth)
b_mid_length = pythag(height_in / math.tan(angle), 6 - 2 * bottom_depth)
b_lat_length = 6 - 2 * bottom_depth
b_out_length = pythag(len_in - 2 * height_in / math.tan(angle), 6 - 2 * bottom_depth)
lateral_thick = 0.18750

print(f"\nLateral lengths (in):")
print(f"  t_lat_length: {t_lat_length:.8f}")
print(f"  t_mid_length: {t_mid_length:.8f}")
print(f"  b_mid_length: {b_mid_length:.8f}")
print(f"  b_lat_length: {b_lat_length:.8f}")
print(f"  b_out_length: {b_out_length:.8f}")

v_t_lat = 3 * (lateral_thick * t_lat_length) * 3 / 16
v_t_mid = 2 * (lateral_thick * t_mid_length) * 3 / 16
v_b_mid = 2 * (lateral_thick * b_mid_length) * 3 / 16
v_b_lat = 3 * (lateral_thick * b_lat_length) * 3 / 16
v_b_out = 2 * (lateral_thick * b_out_length) * 3 / 16

print(f"\nLateral volumes (in^3):")
print(f"  t_lat: {v_t_lat:.8f}")
print(f"  t_mid: {v_t_mid:.8f}")
print(f"  b_mid: {v_b_mid:.8f}")
print(f"  b_lat: {v_b_lat:.8f}")
print(f"  b_out: {v_b_out:.8f}")

lateral_volume_in = v_t_lat + v_t_mid + v_b_mid + v_b_lat + v_b_out

print(f"  TOTAL: {lateral_volume_in:.8f} in^3")

total_volume_in = main_volume_in + lateral_volume_in

print(f"\nTOTAL VOLUME: {total_volume_in:.8f} in^3")
print("\n" + "=" * 67)
print("TORCH CALCULATION (meters)")
print("=" * 67)

material = OchromaWithEpoxy()
angle_m = angle
height_m = inches_to_meters(height_in)
len_m = inches_to_meters(len_in)
incline_length_m = height_m / math.sin(angle_m)
diagonal_length_m = height_m / math.cos(phi)
top_length_m = len_m - 2 * height_m / math.tan(angle_m)
bottom_length_m = len_m
vert_length_m = height_m - inches_to_meters(bottom_thick) - inches_to_meters(top_thick)

print(f"incline_length:  {incline_length_m:.8f} m")
print(f"diagonal_length: {diagonal_length_m:.8f} m")
print(f"top_length:      {top_length_m:.8f} m")
print(f"bottom_length:   {bottom_length_m:.8f} m")
print(f"vert_length:     {vert_length_m:.8f} m")

v_incline_m = 4 * (
    inches_to_meters(incline_thick) * incline_length_m * inches_to_meters(incline_depth)
)
v_diagonal_m = 4 * (
    inches_to_meters(diagonal_thick) * diagonal_length_m * inches_to_meters(diagonal_depth)
)
v_top_m = 2 * (inches_to_meters(top_thick) * top_length_m * inches_to_meters(top_depth))
v_bottom_m = 2 * (inches_to_meters(bottom_thick) * bottom_length_m * inches_to_meters(bottom_depth))
v_mid_vert_m = 2 * (
    inches_to_meters(mid_vert_thick) * vert_length_m * inches_to_meters(mid_vert_depth)
)
v_side_vert_m = 4 * (
    inches_to_meters(side_vert_thick) * vert_length_m * inches_to_meters(side_vert_depth)
)

print(f"\nMain member volumes (m^3):")
print(f"  incline:   {v_incline_m:.12f}")
print(f"  diagonal:  {v_diagonal_m:.12f}")
print(f"  top:       {v_top_m:.12f}")
print(f"  bottom:    {v_bottom_m:.12f}")
print(f"  mid_vert:  {v_mid_vert_m:.12f}")
print(f"  side_vert: {v_side_vert_m:.12f}")

main_volume_m = v_incline_m + v_diagonal_m + v_top_m + v_bottom_m + v_mid_vert_m + v_side_vert_m

print(f"  TOTAL:     {main_volume_m:.12f} m^3")
print(f"  (in in^3:   {main_volume_m / 0.0000163871:.8f} in^3)")

lateral_spacing_m = inches_to_meters(6)
top_depth_m = inches_to_meters(top_depth)
bottom_depth_m = inches_to_meters(bottom_depth)
t_lat_length_m = lateral_spacing_m - 2 * top_depth_m
t_mid_length_m = math.sqrt(
    (height_m / math.tan(angle_m)) ** 2 + (lateral_spacing_m - 2 * top_depth_m) ** 2
)
b_mid_length_m = math.sqrt(
    (height_m / math.tan(angle_m)) ** 2 + (lateral_spacing_m - 2 * bottom_depth_m) ** 2
)
b_lat_length_m = lateral_spacing_m - 2 * bottom_depth_m
b_out_length_m = math.sqrt(
    (len_m - 2 * height_m / math.tan(angle_m)) ** 2 + (lateral_spacing_m - 2 * bottom_depth_m) ** 2
)
lateral_thick_m = 3.0 / 16.0 * 0.0254
lateral_depth_m = 3.0 / 16.0 * 0.0254

print(f"\nLateral lengths (m):")
print(f"  t_lat_length: {t_lat_length_m:.8f}")
print(f"  t_mid_length: {t_mid_length_m:.8f}")
print(f"  b_mid_length: {b_mid_length_m:.8f}")
print(f"  b_lat_length: {b_lat_length_m:.8f}")
print(f"  b_out_length: {b_out_length_m:.8f}")

v_t_lat_m = 3 * (lateral_thick_m * t_lat_length_m * lateral_depth_m)
v_t_mid_m = 2 * (lateral_thick_m * t_mid_length_m * lateral_depth_m)
v_b_mid_m = 2 * (lateral_thick_m * b_mid_length_m * lateral_depth_m)
v_b_lat_m = 3 * (lateral_thick_m * b_lat_length_m * lateral_depth_m)
v_b_out_m = 2 * (lateral_thick_m * b_out_length_m * lateral_depth_m)

print(f"\nLateral volumes (m^3):")
print(
    f"  t_lat: {v_t_lat_m:.12f} ({v_t_lat_m / 0.0000163871:.8f} in^3)"
)  # indentation intentional don't fix
print(f"  t_mid: {v_t_mid_m:.12f} ({v_t_mid_m / 0.0000163871:.8f} in^3)")
print(f"  b_mid: {v_b_mid_m:.12f} ({v_b_mid_m / 0.0000163871:.8f} in^3)")
print(f"  b_lat: {v_b_lat_m:.12f} ({v_b_lat_m / 0.0000163871:.8f} in^3)")
print(f"  b_out: {v_b_out_m:.12f} ({v_b_out_m / 0.0000163871:.8f} in^3)")

lateral_volume_m = v_t_lat_m + v_t_mid_m + v_b_mid_m + v_b_lat_m + v_b_out_m

print(f"  TOTAL: {lateral_volume_m:.12f} m^3")
print(f"  (in in^3: {lateral_volume_m / 0.0000163871:.8f} in^3)")

total_volume_m = main_volume_m + lateral_volume_m

print(f"\nTOTAL VOLUME: {total_volume_m:.12f} m^3")
print(f"(in in^3: {total_volume_m / 0.0000163871:.8f} in^3)")
print("\n" + "=" * 67)
print("COMPARISON")
print("=" * 67)
print(f"Manual total volume: {total_volume_in:.8f} in^3")
print(f"Torch total volume:  {total_volume_m / 0.0000163871:.8f} in^3")
print(f"Difference:          {(total_volume_m / 0.0000163871) - total_volume_in:.8f} in^3")
print(
    f"Percent error:       {100*((total_volume_m / 0.0000163871) - total_volume_in)/total_volume_in:.4f}%"
)
