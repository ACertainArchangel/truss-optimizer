"""
Truss implementations for the optimizer framework.

This package contains implementations of specific truss types. Each truss type
should be in its own subpackage with:
- A truss class (subclassing BaseTruss)
- Differentiable physics functions for optimization

Currently implemented:
- **pratt/** — Single-panel Pratt truss (reference implementation)

To add a new truss type, create a new subpackage following the pratt/ structure:
    trusses/
    └── your_truss/
        ├── __init__.py      # Exports
        ├── truss.py         # YourTruss class
        └── physics.py       # Differentiable physics for optimization

See the pratt/ subpackage for a complete example.
"""

from truss_optimizer.trusses.pratt import PrattTruss, TrussParameters

__all__ = [
    "PrattTruss",
    "TrussParameters",
]
