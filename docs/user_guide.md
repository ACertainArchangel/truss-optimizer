# User Guide

This guide covers everything you need to get started with the Truss Optimizer framework.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Using the Built-in Pratt Truss](#using-the-built-in-pratt-truss)
5. [Creating Custom Truss Types](#creating-custom-truss-types)
6. [Materials](#materials)
7. [Optimization](#optimization)
8. [Analysis](#analysis)
9. [Troubleshooting](#troubleshooting)

## Installation

### From Source

```bash
git clone https://github.com/ACertainArchangel/truss-optimizer.git
cd truss_optimizer
pip install -e .
```

### Optional Dependencies

```bash
# For visualization
pip install -e ".[viz]"

# For development (pytest, etc.)
pip install -e ".[dev]"

# All optional dependencies
pip install -e ".[all]"
```

## Quick Start

Here's a minimal example to optimize a balsa wood Pratt truss bridge:

```python
from truss_optimizer import BridgeOptimizer, PrattTruss, materials

# Create a bridge with initial parameters
bridge = PrattTruss(
    span=0.47,          # meters
    height=0.15,        # meters
    material=materials.BalsaWood()
)

# Create optimizer
optimizer = BridgeOptimizer(
    bridge=bridge,
    objective='load_to_weight',  # Maximize load-to-weight ratio
)

# Run optimization
result = optimizer.optimize(iterations=5000)

# View results
print(f"Critical Load: {result.critical_load:.2f} N")
print(f"Weight: {result.weight:.4f} N")
print(f"Load/Weight: {result.load_to_weight:.1f}")
```

## Core Concepts

### The Framework Architecture

Truss Optimizer is built around these key concepts:

1. **BaseTruss** — Abstract base class that defines the interface for any truss type
2. **Differentiable Physics** — All structural calculations use PyTorch operations for gradient computation
3. **softmin** — A differentiable approximation of `min()` that allows gradients to flow through failure mode selection
4. **BridgeOptimizer** — Generic optimizer that works with any `BaseTruss` subclass

### Why Differentiable?

Traditional optimization methods (genetic algorithms, simulated annealing) don't use gradient information. They explore the parameter space blindly.

**Gradient-based optimization** computes the derivative of your objective (e.g., load-to-weight ratio) with respect to every parameter. This tells the optimizer exactly which direction to move each parameter to improve the design.

The trick is making structural physics differentiable:
- Standard `min()` has zero gradient for non-minimum elements
- `softmin()` approximates `min()` but gives small gradients to all elements
- This lets the optimizer "see" when failure modes are close to becoming critical

## Using the Built-in Pratt Truss

The `PrattTruss` class is a complete reference implementation you can use directly or study to learn how to create your own truss types.

### Pratt Truss Geometry

```
        ┌─────────── Top Chord ───────────┐
       /│\                               /│\
      / │ \        Incline              / │ \
     /  │  \                           /  │  \
    /   │   \                         /   │   \
   /    │    \                       /    │    \
  / Side│Vert \ Diagonal            / Side│Vert \
 /      │      \                   /      │      \
├───────┼───────┼─────────────────┼───────┼───────┤
        │ Mid   │                         │
        │ Vert  │                         │
└─────────────── Bottom Chord ────────────────────┘
```

**Member types:**
- **Incline** — Angled members at ends (compression)
- **Diagonal** — Internal diagonal members (tension)
- **Top Chord** — Horizontal top member (compression)
- **Bottom Chord** — Horizontal bottom member (tension)
- **Mid Vert** — Center vertical (compression)
- **Side Vert** — Verticals at panel points (compression)

### Parameters

```python
from truss_optimizer import PrattTruss, materials
import math

bridge = PrattTruss(
    # Geometry
    span=0.47,                      # Total horizontal length (m)
    height=0.15,                    # Vertical height (m)
    angle=math.radians(30),         # Incline angle from horizontal
    
    # Member cross-sections (thickness × depth)
    incline_thickness=0.006,
    incline_depth=0.010,
    diagonal_thickness=0.003,
    diagonal_depth=0.010,
    mid_vert_thickness=0.006,
    mid_vert_depth=0.010,
    side_vert_thickness=0.006,
    side_vert_depth=0.010,
    top_thickness=0.006,
    top_depth=0.010,
    bottom_thickness=0.003,
    bottom_depth=0.010,
    
    # Material
    material=materials.BalsaWood(),
)
```

### Failure Modes (14 Total)

The Pratt truss models these failure mechanisms:

| Member | Failure Modes |
|--------|---------------|
| Diagonal (tension) | Rupture |
| Bottom Chord (tension) | Rupture |
| Incline (compression) | Euler buckling (in-plane), Euler buckling (out-of-plane), Combined stress |
| Top Chord (compression) | Euler buckling (in-plane), Euler buckling (out-of-plane), Combined stress |
| Mid Vert (compression) | Euler buckling (in-plane), Euler buckling (out-of-plane), Combined stress |
| Side Vert (compression) | Euler buckling (in-plane), Euler buckling (out-of-plane), Combined stress |

## Creating Custom Truss Types

To create your own truss type, subclass `BaseTruss` and implement the required methods.

### Minimum Required Methods

```python
from truss_optimizer import BaseTruss, softmin, euler_buckling_load, DTYPE
import torch

class MyTruss(BaseTruss):
    
    def _define_geometry(self):
        """Called during __init__ to set up geometry."""
        pass
    
    def get_failure_modes(self):
        """Return list of FailureMode objects."""
        pass
    
    def compute_max_load_torch(self, **params):
        """Return (max_load_tensor, governing_mode_name)."""
        pass
    
    def compute_volume_torch(self, **params):
        """Return volume as torch tensor."""
        pass
    
    @property
    def critical_load(self):
        """Maximum load before failure."""
        pass
    
    @property
    def volume(self):
        """Total material volume."""
        pass
```

### Example: Simple Beam Truss

Here's a complete example of a simple truss with just top/bottom chords and diagonals:

```python
from truss_optimizer import BaseTruss, FailureMode, softmin, euler_buckling_load, DTYPE
import torch
import math

class SimpleBeamTruss(BaseTruss):
    """A minimal truss with two chords and diagonal bracing."""
    
    def __init__(self, span, height, material,
                 chord_thickness=0.01, chord_depth=0.02,
                 diagonal_thickness=0.005, diagonal_depth=0.01):
        self.span = span
        self.height = height
        self.material = material
        self.params = {
            'chord_thickness': chord_thickness,
            'chord_depth': chord_depth,
            'diagonal_thickness': diagonal_thickness,
            'diagonal_depth': diagonal_depth,
        }
        self._define_geometry()
    
    def _define_geometry(self):
        """Compute derived geometry."""
        self.diagonal_length = math.sqrt(self.span**2 / 4 + self.height**2)
        self.diagonal_angle = math.atan(self.height / (self.span / 2))
    
    def get_failure_modes(self):
        """Compute all failure modes."""
        modes = []
        
        # Get current max loads
        p = self.params
        E = self.material.E
        sigma_t = self.material.sigma_tension
        sigma_c = self.material.sigma_compression
        
        # Bottom chord tension
        A_bottom = p['chord_thickness'] * p['chord_depth']
        F_bottom = sigma_t * A_bottom * 4 * self.height / self.span
        modes.append(FailureMode(
            name='bottom_chord_rupture',
            member='bottom_chord',
            mechanism='tension',
            max_load=F_bottom
        ))
        
        # Top chord buckling
        A_top = p['chord_thickness'] * p['chord_depth']
        I_top = p['chord_thickness'] * p['chord_depth']**3 / 12
        F_euler = math.pi**2 * E * I_top / (0.5 * self.span / 2)**2
        force_coeff = self.span / (4 * self.height)
        F_top = F_euler / force_coeff
        modes.append(FailureMode(
            name='top_chord_buckling',
            member='top_chord', 
            mechanism='buckling',
            max_load=F_top
        ))
        
        return modes
    
    def compute_max_load_torch(self, **params):
        """Differentiable max load calculation."""
        # Merge with defaults
        p = {**self.params, **params}
        
        # Convert to tensors
        chord_t = torch.as_tensor(p['chord_thickness'], dtype=DTYPE)
        chord_d = torch.as_tensor(p['chord_depth'], dtype=DTYPE)
        E = torch.as_tensor(self.material.E, dtype=DTYPE)
        sigma_t = torch.as_tensor(self.material.sigma_tension, dtype=DTYPE)
        
        # Cross-section properties
        A = chord_t * chord_d
        I = chord_t * chord_d**3 / 12
        
        # Failure mode 1: Bottom chord tension
        F_tension = sigma_t * A * 4 * self.height / self.span
        
        # Failure mode 2: Top chord buckling  
        F_euler = euler_buckling_load(E, I, K=0.5, L=torch.tensor(self.span/2, dtype=DTYPE))
        force_coeff = self.span / (4 * self.height)
        F_buckling = F_euler / force_coeff
        
        # Differentiable minimum
        all_modes = torch.stack([F_tension, F_buckling])
        max_load = softmin(all_modes, alpha=100.0)
        
        # Governing mode (for reporting)
        mode_idx = torch.argmin(all_modes).item()
        mode_names = ['bottom_chord_rupture', 'top_chord_buckling']
        
        return max_load, mode_names[mode_idx]
    
    def compute_volume_torch(self, **params):
        """Differentiable volume calculation."""
        p = {**self.params, **params}
        
        chord_t = torch.as_tensor(p['chord_thickness'], dtype=DTYPE)
        chord_d = torch.as_tensor(p['chord_depth'], dtype=DTYPE)
        diag_t = torch.as_tensor(p['diagonal_thickness'], dtype=DTYPE)
        diag_d = torch.as_tensor(p['diagonal_depth'], dtype=DTYPE)
        
        # Two chords
        chord_vol = 2 * self.span * chord_t * chord_d
        
        # Four diagonals (2 X-braces)
        diag_vol = 4 * self.diagonal_length * diag_t * diag_d
        
        return chord_vol + diag_vol
    
    @property
    def critical_load(self):
        max_load, _ = self.compute_max_load_torch()
        return max_load.item()
    
    @property
    def volume(self):
        return self.compute_volume_torch().item()
    
    def get_optimizable_params(self):
        return self.params.copy()
```

### Key Points for Custom Trusses

1. **Use `torch.as_tensor()` for parameters** — This allows them to be either floats or tensors with gradients

2. **Use `softmin()` instead of `min()`** — This is crucial for gradient flow

3. **Keep computations differentiable** — Avoid `if` statements based on tensor values; use `torch.where()` instead

4. **Return tensors from torch methods** — Even if computing scalars, return as 0-d tensors

5. **Accept `**kwargs` in physics functions** — All `compute_max_load_torch()`, `compute_volume_torch()`, and similar physics functions should accept `**kwargs` to gracefully handle extra parameters. This is critical for compatibility with the optimizer, which may pass all parameters uniformly across different truss types. For example:
   ```python
   def compute_max_load_torch(self, **params):
       # Extract only the parameters you need
       p = {**self.params, **params}
       chord_t = torch.as_tensor(p['chord_thickness'], dtype=DTYPE)
       # Any extra params in **params are safely ignored
   ```

## Materials

### Built-in Materials

```python
from truss_optimizer import materials

# Structural materials
steel = materials.Steel()           # E=200 GPa
aluminum = materials.Aluminum()     # E=70 GPa
titanium = materials.Titanium()     # E=116 GPa

# Wood
balsa = materials.BalsaWood()       # E=3.5 GPa, ρ=200 kg/m³
balsa_light = materials.BalsaWoodLight()  # Lower density variant
balsa_stiff = materials.BalsaWoodStiff()  # Higher E variant

# 3D printing
pla = materials.PLA()
carbon_pla = materials.CarbonFiberPLA()

# Custom
my_material = materials.Custom(
    E=5e9,                    # Young's modulus (Pa)
    sigma_compression=15e6,   # Compressive strength (Pa)
    sigma_tension=25e6,       # Tensile strength (Pa)
    density=180,              # Density (kg/m³)
    name="My Special Balsa"
)
```

### Material Properties

All materials provide:
- `E` — Young's modulus (Pa)
- `sigma_compression` — Compressive strength (Pa)
- `sigma_tension` — Tensile strength (Pa)
- `density` — Mass density (kg/m³)

## Optimization

### Basic Optimization

```python
from truss_optimizer import BridgeOptimizer, PrattTruss, materials

bridge = PrattTruss(span=0.5, height=0.15, material=materials.BalsaWood())

optimizer = BridgeOptimizer(
    bridge=bridge,
    objective='load_to_weight',  # 'max_load', 'min_weight', or 'load_to_weight'
    learning_rate=0.001,
)

result = optimizer.optimize(iterations=5000, verbose=True)
```

### With Constraints

```python
from truss_optimizer.utils import inches_to_meters

optimizer = BridgeOptimizer(
    bridge=bridge,
    constraints={
        'span': (0.47, 0.47),                    # Fixed span
        'height': (0.10, 0.25),                  # Height range
        'incline_thickness': (0.003, 0.013),     # Thickness bounds
    }
)
```

### Multiple Trials

For global optimization, run multiple trials with random starting points:

```python
from truss_optimizer import run_optimization_trials

results = run_optimization_trials(
    n_trials=50,
    bridge_class=PrattTruss,
    bridge_kwargs={'span': 0.47, 'material': materials.BalsaWood()},
    iterations=3000,
)

best = results.best()
print(f"Best load/weight: {best.load_to_weight:.1f}")
```

## Analysis

### Failure Mode Analysis

```python
from truss_optimizer import FailureAnalyzer

bridge = PrattTruss(span=0.47, height=0.15, material=materials.BalsaWood())
analyzer = FailureAnalyzer(bridge)

# Print detailed report
print(analyzer.format_report())

# Get safety factors
safety = analyzer.safety_factors(applied_load=100)  # 100 N applied
```

### Accessing Failure Modes

```python
failure_modes = bridge.get_failure_modes()
for mode in failure_modes:
    print(f"{mode.name}: {mode.max_load:.1f} N")

# Governing mode
print(f"Governing: {bridge.governing_failure_mode}")
print(f"Critical load: {bridge.critical_load:.2f} N")
```

## Troubleshooting

### Optimization Not Converging

1. **Try different learning rates**: Start with 0.001, try 0.01 or 0.0001
2. **Run more iterations**: Some designs need 10,000+ iterations
3. **Use multiple trials**: Global optimization helps escape local minima
4. **Check constraints**: Very tight constraints limit the search space

### NaN or Inf Values

This usually means:
1. **Division by zero** — Check for zero thicknesses
2. **Numerical overflow** — Very large stiffness values
3. **Invalid geometry** — Negative lengths or angles

Add bounds to prevent invalid configurations:
```python
constraints={
    'thickness': (0.001, 0.1),  # Minimum 1mm thickness
}
```

### Gradient Issues

If optimization doesn't improve:
1. Verify your `compute_max_load_torch` uses only differentiable operations
2. Check that you're using `softmin()` instead of `min()`
3. Ensure parameters are converted with `torch.as_tensor()`
