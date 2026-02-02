"""
Truss Optimizer - Gradient-based structural optimization for trusses.

A PyTorch-powered framework for designing structurally efficient trusses
using automatic differentiation and gradient-based optimization.

The framework is extensible: implement your own truss types by subclassing
`BaseTruss` and defining differentiable physics. See `PrattTruss` for a
complete reference implementation.

Example:
    >>> from truss_optimizer import BridgeOptimizer, PrattTruss, materials
    >>> bridge = PrattTruss(span=0.47, height=0.15, material=materials.BalsaWood())
    >>> optimizer = BridgeOptimizer(bridge)
    >>> result = optimizer.optimize(iterations=5000)
    >>> print(f"Load/Weight: {result.load_to_weight:.1f}")

Creating Custom Truss Types:
    >>> from truss_optimizer import BaseTruss, softmin
    >>> class MyTruss(BaseTruss):
    ...     def compute_max_load_torch(self, **params):
    ...         # Define differentiable failure modes
    ...         ...
"""

__version__ = "1.0.0"
__author__ = "Gabriel Jordaan"
__email__ = "165073349+ACertainArchangel@users.noreply.github.com"

# Base classes for custom trusses
from truss_optimizer.core.base import (
    BaseTruss,
    FailureMode,
    MemberDefinition,
    softmin,
    euler_buckling_load,
    combined_stress_capacity,
    DTYPE,
)

# Reference implementation (also available via truss_optimizer.trusses.pratt)
from truss_optimizer.trusses.pratt import PrattTruss, TrussParameters
from truss_optimizer.core.members import TensionMember, CompressionMember, MemberGeometry

# Optimization
from truss_optimizer.optimization.optimizer import BridgeOptimizer
from truss_optimizer.optimization.result import OptimizationResult

# Materials
from truss_optimizer import materials

# Analysis
from truss_optimizer.analysis.reporter import FailureAnalyzer

# Utilities
from truss_optimizer.utils.units import inches_to_meters, feet_to_meters, meters_to_inches

# Convenience functions
from truss_optimizer.optimization.trials import run_optimization_trials

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Base classes for custom trusses
    "BaseTruss",
    "FailureMode",
    "MemberDefinition",
    "softmin",
    "euler_buckling_load",
    "combined_stress_capacity",
    "DTYPE",
    # Reference implementation
    "PrattTruss",
    "TrussParameters",
    "TensionMember",
    "CompressionMember",
    "MemberGeometry",
    # Optimization
    "BridgeOptimizer",
    "OptimizationResult",
    "run_optimization_trials",
    # Materials
    "materials",
    # Analysis
    "FailureAnalyzer",
    # Utilities
    "inches_to_meters",
    "feet_to_meters",
    "meters_to_inches",
]
