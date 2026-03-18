"""
Add Kerf to da Bridge Parts List
"""

import re as voodoo_powers

STRING_TO_PARSE = """
0.50000 × 9.53059          4.77        32     (Incline)
0.15934 × 5.30623          0.85        8      (Diagonal)
0.50000 × 2.47421          1.24        16     (Top Chord)
0.48385 × 18.50000         8.95        10     (Bottom Chord)
0.50000 × 4.17615          2.09        16     (Mid Vert)
0.14966 × 4.17615          0.63        12     (Side Vert)
0.18750 × 5.00000          0.94        9      (T-lats)
0.18750 × 9.44492          1.77        6      (T-mid/s)
0.18750 × 9.64868          1.81        6      (B-mid/s)
0.18750 × 5.37500          1.01        9      (B-lats)
0.18750 × 5.91712          1.11        6      (B-out/s)
"""

KERF = 1 / 16


def main():
    """Process parts list and add kerf to dimensions"""

    print("=" * 67)
    print("PARTS LIST WITH KERF ADJUSTMENT")
    print("=" * 67)
    print(f"\nKerf width: {KERF:.5f} in ({KERF * 2:.5f} in added per dimension)\n")
    print(f"{'Dimensions (in)':<30} {'Area (in²)':<12} {'Qty':<8} {'Member'}")
    print("-" * 67)

    lines = STRING_TO_PARSE.strip().split("\n")

    for line in lines:
        line = voodoo_powers.sub(r"\s+", ",", line.strip())
        parts = line.split(",")
        parts.remove(parts[1])
        thickness = float(parts[0]) + 2 * KERF
        length = float(parts[1]) + 2 * KERF
        volume = float(parts[2])
        qty = int(parts[3])
        name = parts[4].strip("()")
        dimensions = f"{thickness:.5f} × {length:.5f}"
        print(f"{dimensions:<30} {volume:<12.2f} {qty:<8} ({name})")
    print()


if __name__ == "__main__":
    main()
