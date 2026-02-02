"""
Unit conversion utilities.

Provides convenient functions for converting between metric and imperial units
commonly used in structural engineering.
"""

import math


# Length conversions
def inches_to_meters(inches: float) -> float:
    """Convert inches to meters."""
    return inches * 0.0254


def meters_to_inches(meters: float) -> float:
    """Convert meters to inches."""
    return meters / 0.0254


def feet_to_meters(feet: float) -> float:
    """Convert feet to meters."""
    return feet * 0.3048


def meters_to_feet(meters: float) -> float:
    """Convert meters to feet."""
    return meters / 0.3048


def mm_to_meters(mm: float) -> float:
    """Convert millimeters to meters."""
    return mm / 1000


def meters_to_mm(meters: float) -> float:
    """Convert meters to millimeters."""
    return meters * 1000


def cm_to_meters(cm: float) -> float:
    """Convert centimeters to meters."""
    return cm / 100


def meters_to_cm(meters: float) -> float:
    """Convert meters to centimeters."""
    return meters * 100


# Angle conversions
def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return math.radians(degrees)


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(radians)


# Force conversions
def lbf_to_newtons(lbf: float) -> float:
    """Convert pounds-force to Newtons."""
    return lbf * 4.44822


def newtons_to_lbf(newtons: float) -> float:
    """Convert Newtons to pounds-force."""
    return newtons / 4.44822


def kgf_to_newtons(kgf: float) -> float:
    """Convert kilogram-force to Newtons."""
    return kgf * 9.80665


def newtons_to_kgf(newtons: float) -> float:
    """Convert Newtons to kilogram-force."""
    return newtons / 9.80665


# Pressure/Stress conversions
def psi_to_pascals(psi: float) -> float:
    """Convert PSI to Pascals."""
    return psi * 6894.76


def pascals_to_psi(pascals: float) -> float:
    """Convert Pascals to PSI."""
    return pascals / 6894.76


def mpa_to_pascals(mpa: float) -> float:
    """Convert MPa to Pascals."""
    return mpa * 1e6


def pascals_to_mpa(pascals: float) -> float:
    """Convert Pascals to MPa."""
    return pascals / 1e6


def gpa_to_pascals(gpa: float) -> float:
    """Convert GPa to Pascals."""
    return gpa * 1e9


def pascals_to_gpa(pascals: float) -> float:
    """Convert Pascals to GPa."""
    return pascals / 1e9


# Mass/Density conversions
def lb_per_ft3_to_kg_per_m3(lb_per_ft3: float) -> float:
    """Convert lb/ft³ to kg/m³."""
    return lb_per_ft3 * 16.0185


def kg_per_m3_to_lb_per_ft3(kg_per_m3: float) -> float:
    """Convert kg/m³ to lb/ft³."""
    return kg_per_m3 / 16.0185


# Fractional inch helpers (common in woodworking)
def fraction_inch_to_meters(whole: int, numerator: int, denominator: int) -> float:
    """
    Convert fractional inches to meters.
    
    Example:
        >>> fraction_inch_to_meters(3, 1, 8)  # 3-1/8 inches
        0.079375
    """
    inches = whole + numerator / denominator
    return inches_to_meters(inches)


def meters_to_fraction_inch(meters: float, denominator: int = 16) -> str:
    """
    Convert meters to fractional inches string.
    
    Args:
        meters: Length in meters
        denominator: Fraction denominator (default 16 for 1/16")
        
    Returns:
        String like "3-1/8" or "1/4"
        
    Example:
        >>> meters_to_fraction_inch(0.079375)
        '3-1/8'
    """
    inches = meters_to_inches(meters)
    whole = int(inches)
    fraction = inches - whole
    
    # Round to nearest 1/denominator
    numerator = round(fraction * denominator)
    
    # Simplify fraction
    from math import gcd
    if numerator > 0:
        g = gcd(numerator, denominator)
        numerator //= g
        denom = denominator // g
        
        if whole > 0:
            return f"{whole}-{numerator}/{denom}"
        else:
            return f"{numerator}/{denom}"
    else:
        return str(whole)
