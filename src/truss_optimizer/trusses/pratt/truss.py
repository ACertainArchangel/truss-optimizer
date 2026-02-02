"""
Pratt Truss implementation with full structural analysis.

This module provides the PrattTruss class which models a single-panel Pratt
truss bridge with all structural members and failure mode calculations.

This is an example implementation of the BaseTruss interface, demonstrating
how to create a custom truss type with differentiable physics for optimization.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from truss_optimizer.core.base import (
    BaseTruss,
    FailureMode,
    MemberDefinition,
    DTYPE,
    softmin,
    euler_buckling_load,
    combined_stress_capacity,
)
from truss_optimizer.core.members import (
    CompressionMember,
    MemberGeometry,
    TensionMember,
)
from truss_optimizer.materials import Material


@dataclass
class TrussParameters:
    """
    Parameters defining a Pratt truss geometry and member dimensions.
    
    Attributes:
        angle: Angle of incline members from horizontal (radians)
        height: Vertical height of the truss (m)
        span: Total horizontal span (m)
        incline_thickness: Thickness of incline members (m)
        incline_depth: Depth of incline members (m)
        diagonal_thickness: Thickness of diagonal members (m)
        diagonal_depth: Depth of diagonal members (m)
        mid_vert_thickness: Thickness of middle vertical member (m)
        mid_vert_depth: Depth of middle vertical member (m)
        side_vert_thickness: Thickness of side vertical members (m)
        side_vert_depth: Depth of side vertical members (m)
        top_thickness: Thickness of top chord (m)
        top_depth: Depth of top chord (m)
        bottom_thickness: Thickness of bottom chord (m)
        bottom_depth: Depth of bottom chord (m)
    """
    
    # Geometry
    angle: float = math.radians(30)
    height: float = 0.15
    span: float = 0.47
    
    # Member dimensions (thickness × depth)
    incline_thickness: float = 0.01
    incline_depth: float = 0.02
    diagonal_thickness: float = 0.01
    diagonal_depth: float = 0.02
    mid_vert_thickness: float = 0.01
    mid_vert_depth: float = 0.02
    side_vert_thickness: float = 0.01
    side_vert_depth: float = 0.02
    top_thickness: float = 0.01
    top_depth: float = 0.02
    bottom_thickness: float = 0.01
    bottom_depth: float = 0.02
    
    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "TrussParameters":
        """Create parameters from dictionary."""
        # Handle both 'span' and 'length' for compatibility
        if 'length' in data and 'span' not in data:
            data = dict(data)
            data['span'] = data.pop('length')
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PrattTruss(BaseTruss):
    """
    A single-panel Pratt truss bridge with complete structural analysis.
    
    The Pratt truss configuration has diagonal members in tension and vertical
    members in compression, making it efficient for bridges where the load is
    applied to the bottom chord.
    
    This is an example implementation of BaseTruss, demonstrating how to:
    1. Define geometry from parameters
    2. Compute member forces using statics
    3. Calculate failure modes
    4. Implement differentiable physics for optimization
    
    Geometry:
    ```
            ┌─────────── Top Chord ───────────┐
           /│\\                               /│\\
          / │ \\        Incline              / │ \\
         /  │  \\                           /  │  \\
        /   │   \\                         /   │   \\
       /    │    \\                       /    │    \\
      / Side│Vert \\ Diagonal            / Side│Vert \\
     /      │      \\                   /      │      \\
    ├───────┼───────┼These two diags meet┼───────┼───────┤
                       │
    └─────────────── Bottom Chord ──────────———──────────┘
    Also there is a mid vert from the meeting of the diagonals to the middle of the top chord.
    ```
    
    Args:
        params: Truss geometry parameters (or use keyword arguments)
        material: Material properties
        K: Effective length factor for buckling (default 1.0)
        **kwargs: Individual parameters (alternative to params object)
        
    Example:
        >>> from truss_optimizer import PrattTruss, materials
        >>> bridge = PrattTruss(
        ...     span=0.47,
        ...     height=0.15,
        ...     angle=math.radians(30),
        ...     material=materials.BalsaWood()
        ... )
        >>> print(f"Critical load: {bridge.critical_load:.2f} N")
    """
    
    def __init__(
        self,
        params: Optional[TrussParameters] = None,
        material: Optional[Material] = None,
        K: float = 1.0,
        **kwargs,
    ):
        # Handle parameters
        if params is not None:
            self.params = params
        elif kwargs:
            # Handle 'length' -> 'span' conversion
            if 'length' in kwargs and 'span' not in kwargs:
                kwargs['span'] = kwargs.pop('length')
            self.params = TrussParameters(**kwargs)
        else:
            self.params = TrussParameters()
        
        self.material = material
        self.K = K
        
        # Extract material properties
        if material:
            self.E = material.E
            self.sigma_compression = material.sigma_compression
            self.sigma_tension = material.sigma_tension
            self.density = material.density
        else:
            # Default to steel
            self.E = 200e9
            self.sigma_compression = 250e6
            self.sigma_tension = 400e6
            self.density = 7850.0
        
        # Compute derived geometry
        self._compute_geometry()
        
        # Create structural members
        self._create_members()
    
    def _compute_geometry(self) -> None:
        """Compute derived geometric quantities."""
        p = self.params
        
        # Aliases for clarity
        self.angle = p.angle
        self.height = p.height
        self.span = p.span
        
        # Angle of diagonal member from horizontal
        self.phi = math.atan(
            ((self.span - 2 * self.height / math.tan(self.angle)) / 2) / self.height
        )
        
        # Member lengths
        self.L_incline = self.height / math.sin(self.angle)
        self.L_diagonal = self.height / math.cos(self.phi)
        self.L_top = self.span - 2 * self.height / math.tan(self.angle)
        self.L_bottom = self.span
        self.L_mid_vert = self.height
        self.L_side_vert = self.height
    
    def _define_geometry(self) -> None:
        """Define the truss geometry (required by BaseTruss)."""
        self._compute_geometry()
        self._create_members()
    
    def _create_members(self) -> None:
        """Create all structural members."""
        p = self.params
        
        self.members: Dict[str, Union[TensionMember, CompressionMember]] = {}
        
        # Incline members (compression)
        self.members["incline"] = CompressionMember(
            sigma_compression=self.sigma_compression,
            E=self.E,
            K=self.K,
            geometry=MemberGeometry(
                length=self.L_incline,
                thickness=p.incline_thickness,
                depth=p.incline_depth,
            ),
        )
        
        # Diagonal members (tension)
        self.members["diagonal"] = TensionMember(
            sigma_tension=self.sigma_tension,
            geometry=MemberGeometry(
                length=self.L_diagonal,
                thickness=p.diagonal_thickness,
                depth=p.diagonal_depth,
            ),
        )
        
        # Top chord (compression)
        self.members["top_chord"] = CompressionMember(
            sigma_compression=self.sigma_compression,
            E=self.E,
            K=self.K,
            geometry=MemberGeometry(
                length=self.L_top,
                thickness=p.top_thickness,
                depth=p.top_depth,
            ),
        )
        
        # Bottom chord (tension)
        self.members["bottom_chord"] = TensionMember(
            sigma_tension=self.sigma_tension,
            geometry=MemberGeometry(
                length=self.L_bottom,
                thickness=p.bottom_thickness,
                depth=p.bottom_depth,
            ),
        )
        
        # Middle vertical (compression)
        self.members["mid_vert"] = CompressionMember(
            sigma_compression=self.sigma_compression,
            E=self.E,
            K=self.K,
            geometry=MemberGeometry(
                length=self.L_mid_vert,
                thickness=p.mid_vert_thickness,
                depth=p.mid_vert_depth,
            ),
        )
        
        # Side verticals (compression)
        self.members["side_vert"] = CompressionMember(
            sigma_compression=self.sigma_compression,
            E=self.E,
            K=self.K,
            geometry=MemberGeometry(
                length=self.L_side_vert,
                thickness=p.side_vert_thickness,
                depth=p.side_vert_depth,
            ),
        )
    
    def get_axial_force_functions(self) -> Dict[str, Callable[[float], float]]:
        """
        Get functions mapping applied load F to axial force in each member.
        
        Returns:
            Dict mapping member names to functions F -> axial_force
        """
        axial: Dict[str, Callable[[float], float]] = {}
        
        # Incline: F_axial = F / (2·sin(θ))
        axial["incline"] = lambda F: F / (2 * math.sin(self.angle))
        
        # Diagonal: F_axial = F / (6·cos(φ))
        axial["diagonal"] = lambda F: F / (6 * math.cos(self.phi))
        
        # Top chord: combined from incline and diagonal
        def top_axial(F: float) -> float:
            return (
                axial["incline"](F) * math.cos(self.angle) +
                axial["diagonal"](F) * math.sin(self.phi)
            )
        axial["top_chord"] = top_axial
        axial["bottom_chord"] = top_axial  # Same magnitude
        
        # Vertical members share load proportionally to area
        total_vert_area = (
            2 * self.members["side_vert"].area + 
            self.members["mid_vert"].area
        )
        
        def vert_load_fraction(F: float) -> float:
            # Total force covered by vertical members = F/3
            return F / 3
        
        axial["mid_vert"] = lambda F: (
            self.members["mid_vert"].area / total_vert_area * vert_load_fraction(F)
        )
        axial["side_vert"] = lambda F: (
            self.members["side_vert"].area / total_vert_area * vert_load_fraction(F)
        )
        
        return axial
    
    def get_moment_functions(self) -> Dict[str, Optional[Callable[[float], float]]]:
        """
        Get functions mapping applied load F to bending moment in each member.
        
        Most members have zero moment; only top and bottom chords have moments.
        
        Returns:
            Dict mapping member names to moment functions (or None if zero moment)
        """
        moments: Dict[str, Optional[Callable[[float], float]]] = {}
        
        axial = self.get_axial_force_functions()
        
        # No moments in incline, diagonal, vertical members
        moments["incline"] = None
        moments["diagonal"] = None
        moments["mid_vert"] = None
        moments["side_vert"] = None
        
        # Top chord moment at center
        lever_top = self.span / 2 - self.height / math.tan(self.angle)
        moments["top_chord"] = lambda F: axial["side_vert"](F) * lever_top
        
        # Bottom chord moment (at intermediate point)
        moments["bottom_chord"] = lambda F: -axial["side_vert"](F) * lever_top
        
        return moments
    
    def get_failure_modes(self) -> Dict[str, float]:
        """
        Calculate critical applied load for all 14 failure modes.
        
        Failure modes:
        - Tension members: rupture (2 modes)
        - Compression members: buckling in-plane, buckling out-of-plane,
          combined stress (4 members × 3 modes = 12 modes)
        
        Returns:
            Dict mapping failure mode names to critical loads (N)
        """
        failure_modes: Dict[str, float] = {}
        
        axial_funcs = self.get_axial_force_functions()
        moment_funcs = self.get_moment_functions()
        
        for name, member in self.members.items():
            axial_lambda = axial_funcs[name]
            moment_lambda = moment_funcs.get(name)
            
            if isinstance(member, TensionMember):
                failure_modes[f"{name}_rupture"] = member.find_failure_load(
                    axial_lambda=axial_lambda,
                    moment_lambda=moment_lambda,
                )
            
            elif isinstance(member, CompressionMember):
                # In-plane buckling
                failure_modes[f"{name}_buckle"] = member.find_euler_buckling_F(
                    axial_lambda=axial_lambda,
                    moment_lambda=moment_lambda,
                    in_plane=True,
                )
                
                # Out-of-plane buckling
                failure_modes[f"{name}_buckle_out_of_plane"] = member.find_euler_buckling_F(
                    axial_lambda=axial_lambda,
                    moment_lambda=moment_lambda,
                    in_plane=False,
                )
                
                # Combined stress (material crushing)
                failure_modes[f"{name}_combined_stress"] = member.find_material_strength_F(
                    axial_lambda=axial_lambda,
                    moment_lambda=moment_lambda,
                    in_plane=True,
                )
        
        return failure_modes
    
    @property
    def critical_load(self) -> float:
        """
        The critical applied load causing first failure (N).
        
        This is the minimum across all failure modes.
        """
        failure_modes = self.get_failure_modes()
        valid_loads = [v for v in failure_modes.values() if v is not None and v > 0]
        return min(valid_loads) if valid_loads else 0.0
    
    @property
    def governing_failure_mode(self) -> str:
        """The failure mode that governs (has lowest critical load)."""
        failure_modes = self.get_failure_modes()
        return min(failure_modes, key=lambda k: failure_modes[k] if failure_modes[k] else float('inf'))
    
    @property
    def volume(self) -> float:
        """Total volume of all truss members (m³)."""
        return (
            2 * self.members["incline"].volume +
            2 * self.members["diagonal"].volume +
            self.members["top_chord"].volume +
            self.members["bottom_chord"].volume +
            self.members["mid_vert"].volume +
            2 * self.members["side_vert"].volume
        )
    
    @property
    def weight(self) -> float:
        """Total weight of the truss (N)."""
        return self.volume * self.density * 9.81
    
    @property
    def load_to_weight(self) -> float:
        """Load-to-weight ratio (dimensionless)."""
        return self.critical_load / self.weight if self.weight > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Export truss configuration to dictionary."""
        return {
            **self.params.to_dict(),
            "E": self.E,
            "sigma_compression": self.sigma_compression,
            "sigma_tension": self.sigma_tension,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], material: Optional[Material] = None) -> "PrattTruss":
        """Create truss from dictionary."""
        params = TrussParameters.from_dict(data)
        return cls(params=params, material=material)
    
    @classmethod
    def from_file(cls, path: Union[str, Path], material: Optional[Material] = None) -> "PrattTruss":
        """Load truss configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, material=material)
    
    def to_file(self, path: Union[str, Path]) -> None:
        """Save truss configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def __repr__(self) -> str:
        return (
            f"PrattTruss(span={self.span:.3f}m, height={self.height:.3f}m, "
            f"angle={math.degrees(self.angle):.1f}°, material={self.material})"
        )
    
    # =========================================================================
    # BaseTruss interface - Differentiable methods for optimization
    # =========================================================================
    
    def get_optimizable_params(self) -> Dict[str, float]:
        """Return parameters that can be optimized."""
        return self.params.to_dict()
    
    def get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return default bounds for optimizable parameters."""
        return {
            'angle': (math.radians(15), math.radians(60)),
            'height': (0.05, 0.30),
            'span': (0.20, 1.00),
            'incline_thickness': (0.001, 0.05),
            'incline_depth': (0.001, 0.10),
            'diagonal_thickness': (0.001, 0.05),
            'diagonal_depth': (0.001, 0.10),
            'mid_vert_thickness': (0.001, 0.05),
            'mid_vert_depth': (0.001, 0.10),
            'side_vert_thickness': (0.001, 0.05),
            'side_vert_depth': (0.001, 0.10),
            'top_thickness': (0.001, 0.05),
            'top_depth': (0.001, 0.10),
            'bottom_thickness': (0.001, 0.05),
            'bottom_depth': (0.001, 0.10),
        }
    
    def compute_max_load_torch(self, **params) -> Tuple[torch.Tensor, str]:
        """
        Compute maximum load using differentiable PyTorch operations.
        
        This enables gradient-based optimization of all truss parameters.
        Uses softmin for differentiable minimum across failure modes.
        
        Args:
            **params: Truss parameters (can be torch tensors for gradients)
            
        Returns:
            Tuple of (max_load tensor, governing failure mode name)
        """
        # Import from the pratt-specific physics module
        from truss_optimizer.trusses.pratt.physics import compute_max_load
        
        # Merge with material properties
        full_params = {
            'E': self.E,
            'sigma_compression': self.sigma_compression,
            'sigma_tension': self.sigma_tension,
            **self.params.to_dict(),
            **params,  # Override with provided params
        }
        
        return compute_max_load(**full_params)
    
    def compute_volume_torch(self, **params) -> torch.Tensor:
        """
        Compute total material volume using differentiable operations.
        
        Args:
            **params: Truss parameters (can be torch tensors for gradients)
            
        Returns:
            Volume in m³ as a torch tensor
        """
        from truss_optimizer.trusses.pratt.physics import compute_volume
        
        # Merge with current params
        full_params = {
            **self.params.to_dict(),
            **params,
        }
        
        # Remove material properties (not needed for volume)
        for key in ['E', 'sigma_compression', 'sigma_tension']:
            full_params.pop(key, None)
        
        return compute_volume(**full_params)
