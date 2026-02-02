"""
Differentiable structural physics for Pratt trusses.

This module implements the Pratt truss-specific structural physics calculations
in a fully differentiable manner using PyTorch, enabling gradient-based optimization.

Note: The generic helper functions (softmin, euler_buckling_load, etc.) are in
truss_optimizer.core.base. This module contains only Pratt-specific calculations.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from truss_optimizer.core.base import DTYPE, softmin


def compute_max_load(
    angle: torch.Tensor,
    height: torch.Tensor,
    length: torch.Tensor,
    incline_thickness: torch.Tensor,
    diagonal_thickness: torch.Tensor,
    mid_vert_thickness: torch.Tensor,
    side_vert_thickness: torch.Tensor,
    top_thickness: torch.Tensor,
    bottom_thickness: torch.Tensor,
    incline_depth: torch.Tensor,
    diagonal_depth: torch.Tensor,
    mid_vert_depth: torch.Tensor,
    side_vert_depth: torch.Tensor,
    top_depth: torch.Tensor,
    bottom_depth: torch.Tensor,
    E: torch.Tensor,
    sigma_compression: torch.Tensor,
    sigma_tension: torch.Tensor,
    K: float = 1.0,
    temperature: float = 1e-6,
    dtype: torch.dtype = DTYPE,
    device: Optional[str] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute differentiable maximum applied load for a Pratt truss.
    
    All failure modes are computed analytically using linear relationships
    between applied load F and member forces/moments.
    
    Args:
        angle: Incline angle from horizontal (radians)
        height: Truss height (m)
        length: Truss span (m)
        *_thickness: Member thicknesses (m)
        *_depth: Member depths (m)
        E: Young's modulus (Pa)
        sigma_compression: Compressive strength (Pa)
        sigma_tension: Tensile strength (Pa)
        K: Effective length factor for buckling
        temperature: Softmin temperature (lower = sharper)
        dtype: Torch dtype
        device: Torch device
        **kwargs: Additional parameters (ignored for compatibility with custom truss types)
        
    Returns:
        (max_load, failure_loads): Maximum safe applied load and dict of all failure mode loads
    """
    def T(x):
        """Convert to tensor if needed."""
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        return torch.tensor(x, dtype=dtype, device=device)
    
    # Convert all inputs to tensors
    angle = T(angle)
    height = T(height)
    length = T(length)
    E = T(E)
    sigma_compression = T(sigma_compression)
    sigma_tension = T(sigma_tension)
    K = T(K)
    
    incline_thickness = T(incline_thickness)
    diagonal_thickness = T(diagonal_thickness)
    mid_vert_thickness = T(mid_vert_thickness)
    side_vert_thickness = T(side_vert_thickness)
    top_thickness = T(top_thickness)
    bottom_thickness = T(bottom_thickness)
    
    incline_depth = T(incline_depth)
    diagonal_depth = T(diagonal_depth)
    mid_vert_depth = T(mid_vert_depth)
    side_vert_depth = T(side_vert_depth)
    top_depth = T(top_depth)
    bottom_depth = T(bottom_depth)
    
    # Cross-sectional areas
    A_incline = incline_thickness * incline_depth
    A_diagonal = diagonal_thickness * diagonal_depth
    A_mid = mid_vert_thickness * mid_vert_depth
    A_side = side_vert_thickness * side_vert_depth
    A_top = top_thickness * top_depth
    A_bottom = bottom_thickness * bottom_depth
    
    # Moments of inertia
    # In-plane: I = depth × thickness³ / 12
    I_incline_in = incline_depth * incline_thickness**3 / 12
    I_diagonal_in = diagonal_depth * diagonal_thickness**3 / 12
    I_mid_in = mid_vert_depth * mid_vert_thickness**3 / 12
    I_side_in = side_vert_depth * side_vert_thickness**3 / 12
    I_top_in = top_depth * top_thickness**3 / 12
    I_bottom_in = bottom_depth * bottom_thickness**3 / 12
    
    # Out-of-plane: I = thickness × depth³ / 12
    I_incline_out = incline_thickness * incline_depth**3 / 12
    I_diagonal_out = diagonal_thickness * diagonal_depth**3 / 12
    I_mid_out = mid_vert_thickness * mid_vert_depth**3 / 12
    I_side_out = side_vert_thickness * side_vert_depth**3 / 12
    I_top_out = top_thickness * top_depth**3 / 12
    I_bottom_out = bottom_thickness * bottom_depth**3 / 12
    
    # Diagonal angle
    phi = torch.atan(((length - 2 * height / torch.tan(angle)) / 2) / height)
    
    # Member lengths
    L_incline = height / torch.sin(angle)
    L_diagonal = height / torch.cos(phi)
    L_top = length - 2 * height / torch.tan(angle)
    L_bottom = length
    L_mid_vert = height
    L_side_vert = height
    
    # Axial force coefficients: axial_force = coefficient × F
    a_incline = 1.0 / (2.0 * torch.sin(angle))
    a_diagonal = 1.0 / (6.0 * torch.cos(phi))
    a_top = a_incline * torch.cos(angle) + a_diagonal * torch.sin(phi)
    a_bottom = a_top
    
    total_vert_area = 2.0 * A_side + A_mid
    a_mid = (A_mid / total_vert_area) * (1.0 / 3.0)
    a_side = (A_side / total_vert_area) * (1.0 / 3.0)
    
    # Moment coefficients: moment = coefficient × F
    lever_top = length / 2.0 - height / torch.tan(angle)
    b_top_center = a_side * lever_top
    b_bottom_2 = -a_side * lever_top
    
    # Failure load calculations
    failure_loads: Dict[str, torch.Tensor] = {}
    
    # Helper: tension failure
    def tension_F_crit(a_coeff, area, b_coeff, c, I, sigma):
        """Tension rupture with optional moment."""
        if b_coeff == 0:
            a_safe = torch.clamp(torch.abs(a_coeff), min=1e-12)
            return (area * sigma) / a_safe
        denom = torch.clamp(
            torch.abs(a_coeff) / area + torch.abs(b_coeff) * c / I, 
            min=1e-12
        )
        return sigma / denom
    
    # Helper: Euler buckling
    def euler_F_crit(E, I, K, L, a_coeff, area, b_coeff=0, c=0):
        """Euler buckling with optional moment interaction."""
        L_safe = torch.clamp(L, min=1e-12)
        F_cr = (math.pi**2 * E * I) / (K * L_safe)**2
        
        a_safe = torch.clamp(torch.abs(a_coeff), min=1e-12)
        
        if isinstance(b_coeff, (int, float)) and b_coeff == 0:
            return F_cr / a_safe
        
        b_abs = torch.abs(b_coeff) if isinstance(b_coeff, torch.Tensor) else torch.tensor(abs(b_coeff), dtype=dtype, device=device)
        denom = torch.clamp(a_safe + b_abs * c * area / I, min=1e-12)
        return F_cr / denom
    
    # Helper: compression crushing
    def compression_F_crit(a_coeff, area, b_coeff, c, I, sigma):
        """Compression failure from combined stress."""
        if b_coeff == 0:
            denom = torch.clamp(torch.abs(a_coeff) / area, min=1e-12)
        else:
            denom = torch.clamp(
                torch.abs(a_coeff) / area + torch.abs(b_coeff) * c / I, 
                min=1e-12
            )
        return sigma / denom
    
    # TENSION MEMBERS
    failure_loads['diagonal_rupture'] = tension_F_crit(
        a_diagonal, A_diagonal, 0, 0, I_diagonal_in, sigma_tension
    )
    
    c_bottom = bottom_thickness / 2.0
    failure_loads['bottom_chord_rupture'] = tension_F_crit(
        a_bottom, A_bottom, b_bottom_2, c_bottom, I_bottom_in, sigma_tension
    )
    
    # COMPRESSION MEMBERS
    # Incline (no moment)
    failure_loads['incline_buckle'] = euler_F_crit(
        E, I_incline_in, K, L_incline, a_incline, A_incline
    )
    failure_loads['incline_buckle_out_of_plane'] = euler_F_crit(
        E, I_incline_out, K, L_incline, a_incline, A_incline
    )
    failure_loads['incline_combined_stress'] = compression_F_crit(
        a_incline, A_incline, 0, 0, I_incline_in, sigma_compression
    )
    
    # Top chord (with moment)
    c_top = top_thickness / 2.0
    failure_loads['top_chord_buckle'] = euler_F_crit(
        E, I_top_in, K, L_top, a_top, A_top, b_top_center, c_top
    )
    failure_loads['top_chord_buckle_out_of_plane'] = euler_F_crit(
        E, I_top_out, K, L_top, a_top, A_top, b_top_center, c_top
    )
    failure_loads['top_chord_combined_stress'] = compression_F_crit(
        a_top, A_top, b_top_center, c_top, I_top_in, sigma_compression
    )
    
    # Mid vertical (no moment)
    failure_loads['mid_vert_buckle'] = euler_F_crit(
        E, I_mid_in, K, L_mid_vert, a_mid, A_mid
    )
    failure_loads['mid_vert_buckle_out_of_plane'] = euler_F_crit(
        E, I_mid_out, K, L_mid_vert, a_mid, A_mid
    )
    failure_loads['mid_vert_combined_stress'] = compression_F_crit(
        a_mid, A_mid, 0, 0, I_mid_in, sigma_compression
    )
    
    # Side vertical (no moment)
    failure_loads['side_vert_buckle'] = euler_F_crit(
        E, I_side_in, K, L_side_vert, a_side, A_side
    )
    failure_loads['side_vert_buckle_out_of_plane'] = euler_F_crit(
        E, I_side_out, K, L_side_vert, a_side, A_side
    )
    failure_loads['side_vert_combined_stress'] = compression_F_crit(
        a_side, A_side, 0, 0, I_side_in, sigma_compression
    )
    
    # Compute smooth minimum
    vals = torch.stack(list(failure_loads.values()))
    # Note: softmin uses 'alpha' parameter (higher = sharper), we convert temperature
    # (lower = sharper) to alpha: alpha = 1/temperature
    alpha = 1.0 / temperature if temperature > 0 else 1e6
    max_load = softmin(vals, alpha=alpha)
    
    return max_load, failure_loads


def compute_volume(
    angle: torch.Tensor,
    height: torch.Tensor,
    length: torch.Tensor,
    incline_thickness: torch.Tensor,
    diagonal_thickness: torch.Tensor,
    mid_vert_thickness: torch.Tensor,
    side_vert_thickness: torch.Tensor,
    top_thickness: torch.Tensor,
    bottom_thickness: torch.Tensor,
    incline_depth: torch.Tensor,
    diagonal_depth: torch.Tensor,
    mid_vert_depth: torch.Tensor,
    side_vert_depth: torch.Tensor,
    top_depth: torch.Tensor,
    bottom_depth: torch.Tensor,
    dtype: torch.dtype = DTYPE,
    device: Optional[str] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Compute total Pratt truss volume (differentiable).
    
    Args:
        Same as compute_max_load
        **kwargs: Additional parameters (ignored for compatibility with custom truss types)
        
    Returns:
        Total volume (m³)
    """
    def T(x):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        return torch.tensor(x, dtype=dtype, device=device)
    
    angle = T(angle)
    height = T(height)
    length = T(length)
    
    # Member lengths
    phi = torch.atan(((length - 2*height/torch.tan(angle))/2) / height)
    L_incline = height / torch.sin(angle)
    L_diagonal = height / torch.cos(phi)
    L_top = length - 2*height/torch.tan(angle)
    L_bottom = length
    L_vert = height - T(top_thickness) - T(bottom_thickness)
    
    # Volume = sum of (area × length) for each member
    volume = (
        2 * T(incline_thickness) * T(incline_depth) * L_incline +
        2 * T(diagonal_thickness) * T(diagonal_depth) * L_diagonal +
        T(top_thickness) * T(top_depth) * L_top +
        T(bottom_thickness) * T(bottom_depth) * L_bottom +
        T(mid_vert_thickness) * T(mid_vert_depth) * L_vert +
        2 * T(side_vert_thickness) * T(side_vert_depth) * L_vert
    )
    
    return volume


def compute_weight(
    density: float,
    fixed_cost: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Compute Pratt truss weight (differentiable).
    
    Args:
        density: Material density (kg/m³)
        fixed_cost: Additional fixed weight (N), e.g., for connectors/epoxy
        **kwargs: Geometric parameters (same as compute_volume)
        
    Returns:
        Total weight (N)
    """
    volume = compute_volume(**kwargs)
    
    device = volume.device
    dtype = volume.dtype
    
    density_t = torch.tensor(density, dtype=dtype, device=device)
    g = torch.tensor(9.81, dtype=dtype, device=device)
    fixed = torch.tensor(fixed_cost, dtype=dtype, device=device)
    
    return volume * density_t * g + fixed
