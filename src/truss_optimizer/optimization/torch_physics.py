"""
Differentiable structural analysis using PyTorch.

This module re-exports Pratt truss physics from its canonical location for
backward compatibility. New code should import from truss_optimizer.trusses.pratt.

.. deprecated::
    Import from truss_optimizer.trusses.pratt.physics instead.
"""

# Re-export from the new location for backward compatibility
from truss_optimizer.trusses.pratt.physics import (
    compute_max_load,
    compute_volume,
    compute_weight,
)
from truss_optimizer.core.base import DTYPE, softmin

__all__ = [
    "DTYPE",
    "softmin",
    "compute_max_load",
    "compute_volume",
    "compute_weight",
]
