"""
Structural member definitions for tension and compression elements.

This module provides classes for modeling structural members in a truss,
including their failure modes and load capacity calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, List

from scipy.optimize import fsolve


@dataclass
class MemberGeometry:
    """Geometric properties of a structural member."""
    
    length: float  # Member length (m)
    thickness: float  # Cross-section thickness (m)
    depth: float  # Cross-section depth (m)
    
    @property
    def area(self) -> float:
        """Cross-sectional area (m²)."""
        return self.thickness * self.depth
    
    @property
    def volume(self) -> float:
        """Volume of the member (m³)."""
        return self.area * self.length
    
    @property
    def I_in_plane(self) -> float:
        """Second moment of area for in-plane bending (m⁴).
        
        I = depth × thickness³ / 12
        """
        return self.depth * self.thickness**3 / 12
    
    @property
    def I_out_of_plane(self) -> float:
        """Second moment of area for out-of-plane bending (m⁴).
        
        I = thickness × depth³ / 12
        """
        return self.thickness * self.depth**3 / 12
    
    @property
    def c(self) -> float:
        """Distance from neutral axis to extreme fiber (m)."""
        return self.thickness / 2


class TensionMember:
    """
    A structural member primarily under tensile loading.
    
    Tension members can fail by rupture when the combined stress from
    axial load and bending moment exceeds the material's tensile strength.
    
    Args:
        sigma_tension: Ultimate tensile strength (Pa)
        geometry: Member geometric properties
        
    Example:
        >>> geom = MemberGeometry(length=0.5, thickness=0.01, depth=0.02)
        >>> member = TensionMember(sigma_tension=20e6, geometry=geom)
        >>> critical_load = member.find_failure_load(axial_lambda=lambda F: F/2)
    """
    
    def __init__(
        self,
        sigma_tension: float,
        geometry: MemberGeometry,
    ):
        self.sigma_tension = sigma_tension
        self.geometry = geometry
        
        # Aliases for backward compatibility
        self.ultimate_tensile_str = sigma_tension
        self.cross_sec_area = geometry.area
        self.area = geometry.area
        self.depth = geometry.depth
        self.length = geometry.length
    
    @property
    def volume(self) -> float:
        """Volume of the member (m³)."""
        return self.geometry.volume
    
    def max_axial_load(self) -> float:
        """Maximum axial force before pure tension failure (N)."""
        return self.geometry.area * self.sigma_tension
    
    def combined_stress_ratio(self, F_axial: float, M_max: float = 0) -> float:
        """
        Compute the ratio of applied stress to ultimate stress.
        
        Args:
            F_axial: Axial tensile force (N)
            M_max: Maximum bending moment (N·m)
            
        Returns:
            Stress ratio (should be < 1 for safe design)
        """
        sigma = F_axial / self.geometry.area
        
        if M_max and self.geometry.depth:
            sigma += abs(M_max) * self.geometry.c / self.geometry.I_in_plane
            
        return sigma / self.sigma_tension
    
    def find_failure_load(
        self,
        axial_lambda: Callable[[float], float],
        moment_lambda: Optional[Callable[[float], float]] = None,
    ) -> float:
        """
        Find the applied load F that causes tension rupture.
        
        Uses the combined stress criterion including bending if a moment
        function is provided.
        
        Args:
            axial_lambda: Function mapping applied load F to member axial force
            moment_lambda: Optional function mapping F to member bending moment
            
        Returns:
            Critical applied load causing failure (N)
        """
        def failure_equation(F: float) -> float:
            F = abs(F)  # Ensure positive
            F_axial = axial_lambda(F)
            M_max = moment_lambda(F) if moment_lambda else 0
            
            sigma = F_axial / self.geometry.area
            if M_max and self.geometry.depth:
                I = self.geometry.I_in_plane
                c = self.geometry.c
                sigma += abs(M_max) * c / I
                
            return sigma / self.sigma_tension - 1.0
        
        # Try multiple initial guesses to handle two-root problems
        initial_guesses = [
            self.max_axial_load(),
            self.max_axial_load() / 2,
            self.max_axial_load() * 2,
            self.max_axial_load() / 10,
        ]
        
        positive_roots: List[float] = []
        for guess in initial_guesses:
            try:
                root = fsolve(failure_equation, guess)[0]
                if root > 1e-6:  # Only accept positive roots
                    positive_roots.append(root)
            except Exception:
                continue
        
        # Return the smallest positive root (most conservative)
        if positive_roots:
            return min(positive_roots)
        
        # Fallback
        return abs(fsolve(failure_equation, self.max_axial_load())[0])


class CompressionMember:
    """
    A structural member primarily under compressive loading.
    
    Compression members can fail by:
    1. Euler buckling (in-plane or out-of-plane)
    2. Material crushing (combined axial + bending stress)
    
    Args:
        sigma_compression: Ultimate compressive strength (Pa)
        E: Young's modulus (Pa)
        K: Effective length factor for buckling (default 1.0 for pinned-pinned)
        geometry: Member geometric properties
        
    Example:
        >>> geom = MemberGeometry(length=0.5, thickness=0.01, depth=0.02)
        >>> member = CompressionMember(
        ...     sigma_compression=12e6,
        ...     E=3.5e9,
        ...     K=0.5,
        ...     geometry=geom
        ... )
        >>> F_buckle = member.euler_buckling_load(in_plane=True)
    """
    
    def __init__(
        self,
        sigma_compression: float,
        E: float,
        K: float,
        geometry: MemberGeometry,
    ):
        self.sigma_compression = sigma_compression
        self.E = E
        self.K = K
        self.geometry = geometry
        
        # Aliases for backward compatibility
        self.sigma_ult = sigma_compression
        self.length = geometry.length
        self.thickness = geometry.thickness
        self.depth = geometry.depth
        self.cross_sec_area = geometry.area
        self.A = geometry.area
        self.area = geometry.area
        self.I_in_plane = geometry.I_in_plane
        self.I_out_of_plane = geometry.I_out_of_plane
        self.c = geometry.c
    
    @property
    def volume(self) -> float:
        """Volume of the member (m³)."""
        return self.geometry.volume
    
    def euler_buckling_load(self, in_plane: bool = True) -> float:
        """
        Compute the Euler critical buckling load.
        
        F_cr = π²EI / (KL)²
        
        Args:
            in_plane: If True, use in-plane moment of inertia; else out-of-plane
            
        Returns:
            Critical buckling load (N)
        """
        I = self.geometry.I_in_plane if in_plane else self.geometry.I_out_of_plane
        return (math.pi**2 * self.E * I) / (self.K * self.geometry.length)**2
    
    def combined_stress_ratio(
        self,
        F_axial: float,
        M_max: float = 0,
        in_plane: bool = True,
    ) -> float:
        """
        Compute the ratio of applied stress to ultimate stress.
        
        Args:
            F_axial: Axial compressive force (N)
            M_max: Maximum bending moment (N·m)
            in_plane: Use in-plane or out-of-plane moment of inertia
            
        Returns:
            Stress ratio (should be < 1 for safe design)
        """
        I = self.geometry.I_in_plane if in_plane else self.geometry.I_out_of_plane
        sigma = F_axial / self.geometry.area + M_max * self.geometry.c / I
        return sigma / self.sigma_compression
    
    def find_euler_buckling_F(
        self,
        axial_lambda: Callable[[float], float],
        moment_lambda: Optional[Callable[[float], float]] = None,
        in_plane: bool = True,
    ) -> float:
        """
        Find the applied load F that causes Euler buckling.
        
        Uses the interaction formula when moments are present:
        F_axial/F_cr + M·c·A/(F_cr·I) ≤ 1
        
        Args:
            axial_lambda: Function mapping applied load F to member axial force
            moment_lambda: Optional function mapping F to member bending moment
            in_plane: Consider in-plane or out-of-plane buckling
            
        Returns:
            Critical applied load causing Euler buckling (N)
        """
        F_cr_member = self.euler_buckling_load(in_plane)
        
        if moment_lambda is None:
            # Simple case: F_axial ≤ F_cr
            return F_cr_member / abs(axial_lambda(1.0))
        
        # With moments: use interaction formula
        def buckling_interaction(F: float) -> float:
            F_axial = axial_lambda(F)
            M_max = moment_lambda(F)
            I = self.geometry.I_in_plane if in_plane else self.geometry.I_out_of_plane
            c = self.geometry.c
            
            ratio = F_axial / F_cr_member + (M_max * c * self.geometry.area) / (F_cr_member * I)
            return ratio - 1.0
        
        initial_guess = F_cr_member / (2.0 * abs(axial_lambda(1.0)))
        try:
            return fsolve(buckling_interaction, initial_guess)[0]
        except Exception:
            return F_cr_member / abs(axial_lambda(1.0))
    
    def find_material_strength_F(
        self,
        axial_lambda: Callable[[float], float],
        moment_lambda: Optional[Callable[[float], float]] = None,
        in_plane: bool = True,
    ) -> float:
        """
        Find the applied load F that causes material crushing failure.
        
        Args:
            axial_lambda: Function mapping applied load F to member axial force
            moment_lambda: Optional function mapping F to member bending moment
            in_plane: Use in-plane or out-of-plane moment of inertia
            
        Returns:
            Critical applied load causing material failure (N)
        """
        def material_failure(F: float) -> float:
            F_axial = axial_lambda(F)
            M_max = moment_lambda(F) if moment_lambda else 0
            
            I = self.geometry.I_in_plane if in_plane else self.geometry.I_out_of_plane
            sigma = F_axial / self.geometry.area
            if M_max:
                sigma += M_max * self.geometry.c / I
                
            return sigma / self.sigma_compression - 1.0
        
        initial_guess = self.geometry.area * self.sigma_compression / abs(axial_lambda(1.0))
        try:
            return fsolve(material_failure, initial_guess)[0]
        except Exception:
            return float('inf')
    
    def find_critical_load(
        self,
        axial_lambda: Callable[[float], float],
        moment_lambda: Optional[Callable[[float], float]] = None,
        in_plane: bool = True,
    ) -> float:
        """
        Find the critical applied load considering all compression failure modes.
        
        Returns the minimum of Euler buckling and material crushing limits.
        
        Args:
            axial_lambda: Function mapping applied load F to member axial force
            moment_lambda: Optional function mapping F to member bending moment
            in_plane: Consider in-plane or out-of-plane failure
            
        Returns:
            Critical applied load (N)
        """
        F_euler = self.find_euler_buckling_F(axial_lambda, moment_lambda, in_plane)
        F_material = self.find_material_strength_F(axial_lambda, moment_lambda, in_plane)
        return min(F_euler, F_material)
