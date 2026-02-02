# Examples

This directory contains example scripts demonstrating the Truss Optimizer framework.

## Running Examples

```bash
cd truss_optimizer
pip install -e .
python examples/01_basic_optimization.py
```

## Example Files

### Using the Built-in Pratt Truss

| File | Description |
|------|-------------|
| [01_basic_optimization.py](01_basic_optimization.py) | Simplest example - optimize a Pratt truss bridge |
| [02_multiple_trials.py](02_multiple_trials.py) | Run multiple trials to find global optimum |
| [03_material_comparison.py](03_material_comparison.py) | Compare different materials |
| [04_failure_analysis.py](04_failure_analysis.py) | Detailed analysis of failure modes |
| [05_competition_setup.py](05_competition_setup.py) | Full setup for bridge-building competition |

### Creating Custom Truss Types

| File | Description |
|------|-------------|
| [06_custom_truss_type.py](06_custom_truss_type.py) | **How to create your own truss type** - demonstrates subclassing `BaseTruss` |

## Quick Reference

### Basic Optimization

```python
from truss_optimizer import BridgeOptimizer, PrattTruss, materials

bridge = PrattTruss(span=0.47, height=0.15, material=materials.BalsaWood())
optimizer = BridgeOptimizer(bridge, objective='load_to_weight')
result = optimizer.optimize(iterations=5000)
```

### Custom Truss Type (Minimal)

```python
from truss_optimizer import BaseTruss, softmin, DTYPE
import torch

class MyTruss(BaseTruss):
    def compute_max_load_torch(self, **params):
        # Compute failure loads using torch operations
        loads = torch.stack([F_mode1, F_mode2, F_mode3])
        return softmin(loads), 'governing_mode_name'
    
    def compute_volume_torch(self, **params):
        return total_volume_tensor
    
    # Plus: _define_geometry, critical_load, volume, get_failure_modes
```

See [06_custom_truss_type.py](06_custom_truss_type.py) for a complete working example.

## Tips

- Start with `01_basic_optimization.py` to verify installation
- Use `02_multiple_trials.py` for production-quality designs  
- Study `06_custom_truss_type.py` to learn how to create your own truss types
- Always validate designs with `04_failure_analysis.py` before building
