"""
Material property definitions for structural analysis.

This module provides a library of common structural materials with their
mechanical properties, as well as the ability to define custom materials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Material:
    """
    Base class for material properties.
    
    Attributes:
        E: Young's modulus (Pa) - stiffness
        sigma_compression: Ultimate compressive strength (Pa)
        sigma_tension: Ultimate tensile strength (Pa)
        density: Material density (kg/m³)
        name: Human-readable material name
        
    Example:
        >>> steel = Material(
        ...     E=200e9,
        ...     sigma_compression=250e6,
        ...     sigma_tension=400e6,
        ...     density=7850.0,
        ...     name="Structural Steel"
        ... )
    """
    
    E: float  # Young's modulus (Pa)
    sigma_compression: float  # Compressive strength (Pa)
    sigma_tension: float  # Tensile strength (Pa)
    density: float  # Density (kg/m³)
    name: Optional[str] = None
    
    def __repr__(self) -> str:
        if self.name:
            return f"{self.name}"
        return f"Material(E={self.E:.2e}, σ_c={self.sigma_compression:.2e}, σ_t={self.sigma_tension:.2e})"


# ============================================================================
# Pre-defined Materials
# ============================================================================

class Steel(Material):
    """
    Structural steel (A36/S275).
    
    Properties:
        - E: 200 GPa
        - σ_compression: 250 MPa
        - σ_tension: 400 MPa
        - density: 7,850 kg/m³
    """
    
    def __init__(self):
        super().__init__(
            E=200e9,
            sigma_compression=250e6,
            sigma_tension=400e6,
            density=7850.0,
            name="Steel",
        )


class Aluminum(Material):
    """
    Aluminum alloy (6061-T6).
    
    Properties:
        - E: 69 GPa
        - σ_compression: 150 MPa
        - σ_tension: 200 MPa
        - density: 2,700 kg/m³
    """
    
    def __init__(self):
        super().__init__(
            E=69e9,
            sigma_compression=150e6,
            sigma_tension=200e6,
            density=2700.0,
            name="Aluminum",
        )


class Titanium(Material):
    """
    Titanium alloy (Ti-6Al-4V).
    
    Properties:
        - E: 114 GPa
        - σ_compression: 900 MPa
        - σ_tension: 950 MPa
        - density: 4,500 kg/m³
    """
    
    def __init__(self):
        super().__init__(
            E=114e9,
            sigma_compression=900e6,
            sigma_tension=950e6,
            density=4500.0,
            name="Titanium",
        )


class BalsaWood(Material):
    """
    Balsa wood (Ochroma pyramidale) - medium density.
    
    Excellent strength-to-weight ratio, commonly used in model bridges.
    
    Properties:
        - E: 3.7 GPa
        - σ_compression: 12 MPa
        - σ_tension: 20 MPa
        - density: 200 kg/m³
        
    Note:
        Balsa wood properties vary significantly with density.
        Use LightBalsaWood or HeavyBalsaWood for variants.
    """
    
    def __init__(self):
        super().__init__(
            E=3.71e9,
            sigma_compression=11.6e6,
            sigma_tension=19.6e6,
            density=200.0,
            name="Balsa Wood",
        )


class LightBalsaWood(Material):
    """
    Light balsa wood (low density grade).
    
    Properties:
        - E: 3.0 GPa
        - σ_compression: 10 MPa
        - σ_tension: 15 MPa
        - density: 160 kg/m³
    """
    
    def __init__(self):
        super().__init__(
            E=3e9,
            sigma_compression=10e6,
            sigma_tension=15e6,
            density=160.0,
            name="Light Balsa Wood",
        )


class HeavyBalsaWood(Material):
    """
    Heavy balsa wood (high density grade).
    
    Properties:
        - E: 6.0 GPa
        - σ_compression: 20 MPa
        - σ_tension: 25 MPa
        - density: 320 kg/m³
    """
    
    def __init__(self):
        super().__init__(
            E=6e9,
            sigma_compression=20e6,
            sigma_tension=25e6,
            density=320.0,
            name="Heavy Balsa Wood",
        )


class ConservativeBalsaWood(Material):
    """
    Balsa wood with conservative (safe) property estimates.
    
    Use this when material properties are uncertain.
    
    Properties:
        - E: 2.0 GPa (conservative)
        - σ_compression: 2.5 MPa (very conservative)
        - σ_tension: 3.0 MPa (very conservative)
        - density: 240 kg/m³ (upper estimate)
    """
    
    def __init__(self):
        super().__init__(
            E=2.0e9,
            sigma_compression=2.5e6,
            sigma_tension=3.0e6,
            density=240.0,
            name="Conservative Balsa Wood",
        )


class Concrete(Material):
    """
    Standard concrete (C30/37).
    
    Note: Concrete is weak in tension; typically used with steel reinforcement.
    
    Properties:
        - E: 30 GPa
        - σ_compression: 40 MPa
        - σ_tension: 4 MPa
        - density: 2,400 kg/m³
    """
    
    def __init__(self):
        super().__init__(
            E=30e9,
            sigma_compression=40e6,
            sigma_tension=4e6,
            density=2400.0,
            name="Concrete",
        )


class PLA(Material):
    """
    Polylactic Acid (PLA) - common 3D printing material.
    
    Properties:
        - E: 3.5 GPa
        - σ_compression: 60 MPa
        - σ_tension: 50 MPa
        - density: 1,250 kg/m³
    """
    
    def __init__(self):
        super().__init__(
            E=3.5e9,
            sigma_compression=60e6,
            sigma_tension=50e6,
            density=1250.0,
            name="PLA",
        )


class CarbonFiberPLA(Material):
    """
    Carbon fiber reinforced PLA (3D printing composite).
    
    Properties:
        - E: 10 GPa
        - σ_compression: 100 MPa
        - σ_tension: 90 MPa
        - density: 1,350 kg/m³
    """
    
    def __init__(self):
        super().__init__(
            E=10e9,
            sigma_compression=100e6,
            sigma_tension=90e6,
            density=1350.0,
            name="Carbon Fiber PLA",
        )


class Custom(Material):
    """
    Create a custom material with specified properties.
    
    Args:
        E: Young's modulus (Pa)
        sigma_compression: Ultimate compressive strength (Pa)
        sigma_tension: Ultimate tensile strength (Pa)
        density: Material density (kg/m³)
        name: Optional name for the material
        
    Example:
        >>> carbon_fiber = Custom(
        ...     E=150e9,
        ...     sigma_compression=1000e6,
        ...     sigma_tension=2000e6,
        ...     density=1600.0,
        ...     name="Carbon Fiber Composite"
        ... )
    """
    
    def __init__(
        self,
        E: float,
        sigma_compression: float,
        sigma_tension: float,
        density: float,
        name: Optional[str] = "Custom Material",
    ):
        super().__init__(
            E=E,
            sigma_compression=sigma_compression,
            sigma_tension=sigma_tension,
            density=density,
            name=name,
        )


# Convenient aliases
OchromaWood = BalsaWood  # Scientific name for balsa
