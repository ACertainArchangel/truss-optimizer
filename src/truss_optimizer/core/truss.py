"""
Pratt Truss implementation with full structural analysis.

This module re-exports PrattTruss from its canonical location for backward
compatibility. New code should import from truss_optimizer.trusses.pratt.

.. deprecated::
    Import from truss_optimizer.trusses.pratt instead.
"""

# Re-export from the new location for backward compatibility
from truss_optimizer.trusses.pratt.truss import PrattTruss, TrussParameters

__all__ = ["PrattTruss", "TrussParameters"]
