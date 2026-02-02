"""
Base classes for truss optimization.

This module defines the abstract interfaces that all truss implementations
must follow. Users can create custom truss types by subclassing these bases
and implementing the required methods with differentiable PyTorch operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch

# Use float64 for numerical stability in structural calculations
DTYPE = torch.float64


@dataclass
class FailureMode:
    """
    Represents a single failure mode in a structural analysis.
    
    Attributes:
        name: Human-readable identifier (e.g., "top_chord_buckling_in_plane")
        member: Name of the member that fails in this mode
        mechanism: Type of failure ("tension", "compression", "buckling", "combined")
        max_load: Maximum load before failure in this mode (Newtons)
        description: Optional longer description of the failure mode
    """
    name: str
    member: str
    mechanism: str
    max_load: float
    description: str = ""
    
    def __repr__(self) -> str:
        return f"FailureMode({self.name}: {self.max_load:.1f} N)"


@dataclass 
class MemberDefinition:
    """
    Definition of a structural member.
    
    Attributes:
        name: Unique identifier for the member
        length: Length in meters
        area: Cross-sectional area in m²
        I_min: Minimum second moment of area in m⁴ (for buckling)
        I_max: Maximum second moment of area in m⁴
        is_tension: Whether member is primarily in tension
        axial_force_coefficient: Multiplier to convert applied load to member force
        K: Effective length factor for buckling (default 0.5 for fixed-fixed)
    """
    name: str
    length: float
    area: float
    I_min: float
    I_max: float
    is_tension: bool
    axial_force_coefficient: float
    K: float = 0.5


class BaseTruss(ABC):
    """
    Abstract base class for all truss types.
    
    To create a custom truss, subclass this and implement:
    - `_define_geometry()`: Set up member geometries from parameters
    - `_compute_member_forces()`: Define how loads transfer to members
    - `get_failure_modes()`: Return all possible failure modes
    - `compute_max_load_torch()`: Differentiable max load calculation
    - `compute_volume_torch()`: Differentiable volume calculation
    
    Example:
        class MyCustomTruss(BaseTruss):
            def __init__(self, span, height, material, **thicknesses):
                self.span = span
                self.height = height
                self.material = material
                self.thicknesses = thicknesses
                self._define_geometry()
            
            def _define_geometry(self):
                # Define members based on parameters
                ...
            
            def compute_max_load_torch(self, **params) -> Tuple[torch.Tensor, str]:
                # Return (max_load, governing_mode) using differentiable ops
                ...
    """
    
    @abstractmethod
    def _define_geometry(self) -> None:
        """
        Define the truss geometry and create member definitions.
        
        This method should populate `self.members` with MemberDefinition
        objects based on the truss parameters.
        """
        pass
    
    @abstractmethod
    def get_failure_modes(self) -> List[FailureMode]:
        """
        Compute and return all failure modes for the current geometry.
        
        Returns:
            List of FailureMode objects, one for each way the truss can fail.
        """
        pass
    
    @abstractmethod
    def compute_max_load_torch(self, **params) -> Tuple[torch.Tensor, str]:
        """
        Compute maximum load using differentiable PyTorch operations.
        
        This is the core method that enables gradient-based optimization.
        All operations must use torch tensors and differentiable functions.
        
        Args:
            **params: Truss parameters as torch tensors or floats
            
        Returns:
            Tuple of (max_load tensor, governing failure mode name)
        """
        pass
    
    @abstractmethod
    def compute_volume_torch(self, **params) -> torch.Tensor:
        """
        Compute total material volume using differentiable operations.
        
        Args:
            **params: Truss parameters as torch tensors or floats
            
        Returns:
            Volume in m³ as a torch tensor
        """
        pass
    
    @property
    @abstractmethod
    def critical_load(self) -> float:
        """Maximum load the truss can support (Newtons)."""
        pass
    
    @property
    @abstractmethod
    def volume(self) -> float:
        """Total material volume (m³)."""
        pass
    
    @property
    def weight(self) -> float:
        """Total weight (kg), requires material with density."""
        if not hasattr(self, 'material') or self.material is None:
            raise ValueError("Material must be set to compute weight")
        return self.volume * self.material.density
    
    @property
    def load_to_weight(self) -> float:
        """Load-to-weight ratio (N/kg)."""
        return self.critical_load / self.weight
    
    def get_optimizable_params(self) -> Dict[str, float]:
        """
        Return parameters that can be optimized.
        
        Override this to specify which parameters the optimizer can tune.
        
        Returns:
            Dict mapping parameter names to current values
        """
        return {}
    
    def get_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Return default bounds for optimizable parameters.
        
        Override this to provide sensible defaults for your truss type.
        
        Returns:
            Dict mapping parameter names to (min, max) tuples
        """
        return {}


def softmin(values: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    """
    Differentiable approximation of minimum using log-sum-exp.
    
    This is crucial for optimization - a hard `min()` has zero gradient
    for non-minimum elements, making optimization difficult. Softmin
    provides smooth gradients that flow through all elements.
    
    Args:
        values: Tensor of values to find minimum of
        alpha: Smoothing parameter (higher = closer to true min, but less smooth)
        
    Returns:
        Approximate minimum as a differentiable tensor
        
    Note:
        For structural optimization, alpha=100 provides a good balance
        between accuracy and gradient smoothness.
    """
    return -torch.logsumexp(-alpha * values, dim=0) / alpha


def euler_buckling_load(E, I, K: float, L) -> torch.Tensor:
    """
    Compute Euler critical buckling load.
    
    F_cr = π²EI / (KL)²
    
    Args:
        E: Young's modulus (Pa) - float or tensor
        I: Second moment of area (m⁴) - float or tensor
        K: Effective length factor - float
        L: Member length (m) - float or tensor
        
    Returns:
        Critical buckling load (N)
    """
    import math
    E = torch.as_tensor(E, dtype=DTYPE)
    I = torch.as_tensor(I, dtype=DTYPE)
    L = torch.as_tensor(L, dtype=DTYPE)
    return (math.pi ** 2 * E * I) / (K * L) ** 2


def combined_stress_capacity(P_euler: torch.Tensor, 
                             P_yield: torch.Tensor) -> torch.Tensor:
    """
    Compute capacity under combined buckling and yielding.
    
    Uses the interaction formula: 1/P_combined = 1/P_euler + 1/P_yield
    
    Args:
        P_euler: Euler buckling capacity (N)
        P_yield: Yield/crushing capacity (N)
        
    Returns:
        Combined capacity (N)
    """
    return 1.0 / (1.0 / P_euler + 1.0 / P_yield)
