"""
Pratt Truss implementation.

This subpackage provides the PrattTruss class and its differentiable physics,
serving as a reference implementation for creating custom truss types.

Modules:
- truss.py — PrattTruss class with structural analysis
- physics.py — Differentiable PyTorch physics for optimization
"""

from truss_optimizer.trusses.pratt.truss import PrattTruss, TrussParameters
from truss_optimizer.trusses.pratt.physics import (
    compute_max_load,
    compute_volume,
    compute_weight,
)

__all__ = [
    "PrattTruss",
    "TrussParameters",
    "compute_max_load",
    "compute_volume",
    "compute_weight",
]
