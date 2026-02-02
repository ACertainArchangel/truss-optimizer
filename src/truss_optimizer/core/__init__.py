"""
Core structural components for truss optimization.

Provides base classes for creating custom truss types, plus a reference
implementation (PrattTruss) demonstrating best practices.
"""

from truss_optimizer.core.base import (
    BaseTruss,
    FailureMode,
    MemberDefinition,
    softmin,
    euler_buckling_load,
    combined_stress_capacity,
    DTYPE,
)
from truss_optimizer.core.members import TensionMember, CompressionMember, MemberGeometry
from truss_optimizer.core.truss import PrattTruss, TrussParameters

__all__ = [
    # Base classes for custom trusses
    "BaseTruss",
    "FailureMode",
    "MemberDefinition",
    # Helper functions for differentiable physics
    "softmin",
    "euler_buckling_load",
    "combined_stress_capacity",
    "DTYPE",
    # Member types
    "TensionMember",
    "CompressionMember",
    "MemberGeometry",
    # Reference implementation
    "PrattTruss",
    "TrussParameters",
]
