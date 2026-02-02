# 🏗️ Truss Optimizer

**An extensible gradient-based optimization framework for structural trusses using PyTorch automatic differentiation.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


## Why did I make this?

For a statics class at the University of Nevada, Reno, we were tasked with designing a model bridge out of balsa wood that had the highest critical load to weight ratio. I was a little frustrated, because we were not given any of the tools or theory to come to an objective conclusion of what we thought was best, so I designed this framework, and built two bridges from the results. The project was very succesful, and the bridges broke the records set across all UNR participants for two recent semesters. The original project was a messy repo with about 20 scattered ad hoc utility files, so I had copilot quickly copy the most important files here, and add user friendly documentation.

## What is This

Truss Optimizer is a **framework for optimizing structural trusses** using gradient-based methods. Instead of trial-and-error or genetic algorithms, it uses PyTorch's automatic differentiation to compute exact gradients through your structural physics, enabling fast convergence to optimal designs.

**Key Insight**: If you can write your truss's failure modes as differentiable functions, this framework will optimize it efficiently.

> **📝 Note on Terminology**: While we know that structures with moment-carrying connections are technically more accurate to call "frames" rather than "trusses," we use "truss" throughout this framework because the classic bridge designs (Warren truss, Pratt truss, etc.) are universally known by these names. So yes, it might be a "Warren frame" in structural analysis terms, but nobody says that :p

### Features

- **🔥 Gradient-Based Optimization** — Uses PyTorch autodiff for efficient parameter tuning
- **🧩 Extensible Design** — Subclass `BaseTruss` to create your own truss types  
- **📐 Built-in Physics** — Helper functions for Euler buckling, combined stress, softmin
- **🛠️ Material Library** — Steel, Aluminum, Balsa, or define custom materials
- **📊 Multi-Objective** — Maximize load, minimize weight, or optimize ratios
- **🎯 Reference Implementation** — Complete `PrattTruss` example to learn from

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ACertainArchangel/truss-optimizer.git
cd truss_optimizer
pip install -e .
```

### Using the Built-in Pratt Truss

```python
from truss_optimizer import BridgeOptimizer, PrattTruss, materials

# Create a Pratt truss bridge
bridge = PrattTruss(
    span=0.47,        # meters
    height=0.15,      # meters  
    material=materials.BalsaWood()
)

# Optimize it
optimizer = BridgeOptimizer(bridge, objective='load_to_weight')
result = optimizer.optimize(iterations=5000)

print(f"Critical Load: {result.critical_load:.2f} N")
print(f"Load/Weight: {result.load_to_weight:.1f}")
```

## 🧩 Creating Custom Truss Types

The real power is creating your own truss implementations. Here's how:

### 1. Subclass `BaseTruss`

```python
from truss_optimizer import BaseTruss, softmin, euler_buckling_load, DTYPE
import torch
import math

class WarrenTruss(BaseTruss):
    """A Warren truss with diagonal members alternating direction."""
    
    def __init__(self, span, height, num_panels, material, **member_dims):
        self.span = span
        self.height = height
        self.num_panels = num_panels
        self.material = material
        self.member_dims = member_dims
        self._define_geometry()
    
    def _define_geometry(self):
        """Calculate member lengths and geometry."""
        self.panel_length = self.span / self.num_panels
        self.diagonal_length = math.sqrt(self.panel_length**2 + self.height**2)
        self.diagonal_angle = math.atan(self.height / self.panel_length)
```

### 2. Implement Differentiable Physics

The key method is `compute_max_load_torch()`. This must be differentiable!

```python
def compute_max_load_torch(self, **params) -> tuple[torch.Tensor, str]:
    """
    Compute max load using ONLY differentiable operations.

    This example is overly simplified for clarity, real truss physics will be more complex and involve things like combined stress checks with moments. Then yes, it's technically a frame, but we just call it a truss because nobody says "warren frame" or "pratt frame".
    
    Returns: (max_load_tensor, governing_mode_name)
    """
    # Get parameters (can be torch tensors for gradient flow)
    chord_t = torch.as_tensor(params.get('chord_thickness', 0.01), dtype=DTYPE)
    chord_d = torch.as_tensor(params.get('chord_depth', 0.02), dtype=DTYPE)
    
    E = torch.as_tensor(self.material.E, dtype=DTYPE)
    sigma_t = torch.as_tensor(self.material.sigma_tension, dtype=DTYPE)
    sigma_c = torch.as_tensor(self.material.sigma_compression, dtype=DTYPE)
    
    # Compute cross-section properties
    A = chord_t * chord_d
    I = chord_t * chord_d**3 / 12
    
    # === Failure Mode 1: Bottom chord tension rupture ===
    # Axial force in bottom chord = M / height (approx)
    # So F_applied that causes rupture:
    F_tension = sigma_t * A * self.height / (self.span / 4)
    
    # === Failure Mode 2: Top chord buckling ===
    F_euler = euler_buckling_load(E, I, K=1.0, L=self.panel_length)
    # Convert to applied load using force coefficient
    force_coeff = self.span / (4 * self.height)  # approximate
    F_buckling = F_euler / force_coeff
    
    # === Combine with softmin (differentiable minimum aprox) ===
    all_modes = torch.stack([F_tension, F_buckling])
    max_load = softmin(all_modes, alpha=100.0)
    
    # Determine governing mode (for reporting only)
    mode_idx = torch.argmin(all_modes).item()
    mode_names = ['bottom_chord_rupture', 'top_chord_buckling']
    
    return max_load, mode_names[mode_idx]
```

### 3. Implement Volume Calculation

```python
def compute_volume_torch(self, **params) -> torch.Tensor:
    """Compute total material volume (differentiable)."""
    chord_t = torch.as_tensor(params.get('chord_thickness', 0.01), dtype=DTYPE)
    chord_d = torch.as_tensor(params.get('chord_depth', 0.02), dtype=DTYPE)
    diag_t = torch.as_tensor(params.get('diagonal_thickness', 0.01), dtype=DTYPE)
    diag_d = torch.as_tensor(params.get('diagonal_depth', 0.02), dtype=DTYPE)
    
    # Top and bottom chords
    chord_vol = 2 * self.span * chord_t * chord_d
    
    # Diagonals (2 per panel for Warren truss)
    num_diagonals = 2 * self.num_panels
    diag_vol = num_diagonals * self.diagonal_length * diag_t * diag_d
    
    return chord_vol + diag_vol
```

### 4. Add Required Properties

```python
@property
def critical_load(self) -> float:
    """Maximum load (calls torch method and extracts value)."""
    max_load, _ = self.compute_max_load_torch(**self.member_dims)
    return max_load.item()

@property  
def volume(self) -> float:
    """Total volume."""
    return self.compute_volume_torch(**self.member_dims).item()

def get_failure_modes(self) -> list:
    """Return list of FailureMode objects for analysis."""
    from truss_optimizer import FailureMode
    # Compute each mode separately for reporting
    ...
    return [FailureMode(...), ...]

def get_optimizable_params(self) -> dict:
    """Which parameters can the optimizer tune?"""
    return {
        'chord_thickness': self.member_dims.get('chord_thickness', 0.01),
        'chord_depth': self.member_dims.get('chord_depth', 0.02),
        ...
    }
```

### 5. Use It!

```python
truss = WarrenTruss(
    span=1.0,
    height=0.2,
    num_panels=4,
    material=materials.Steel(),
    chord_thickness=0.01,
    chord_depth=0.02
)

optimizer = BridgeOptimizer(truss, objective='load_to_weight')
result = optimizer.optimize(iterations=3000)
```

## 🔧 Helper Functions

The framework provides utilities for common structural calculations:

```python
from truss_optimizer import softmin, euler_buckling_load, combined_stress_capacity, DTYPE

# Differentiable minimum (crucial for optimization!)
loads = torch.tensor([100.0, 150.0, 80.0], dtype=DTYPE)
min_load = softmin(loads, alpha=100.0)  # ≈ 80, but with gradients to all elements

# Euler buckling
F_cr = euler_buckling_load(E=200e9, I=1e-8, K=1.0, L=0.5)

# Combined stress (buckling + yielding interaction)
F_combined = combined_stress_capacity(P_euler=1000, P_yield=800)
```

## 📦 Materials

```python
from truss_optimizer import materials

# Built-in
steel = materials.Steel()
aluminum = materials.Aluminum()
balsa = materials.BalsaWood()
pla = materials.PLA()

# Custom
carbon = materials.Custom(
    E=70e9,                     # Young's modulus (Pa)
    sigma_compression=600e6,    # Compressive strength (Pa)
    sigma_tension=700e6,        # Tensile strength (Pa)
    density=1600,               # kg/m³
    name="Carbon Fiber"
)
```

## 📁 Project Structure

```
truss_optimizer/
├── src/truss_optimizer/
│   ├── __init__.py              # Main exports
│   ├── core/
│   │   ├── base.py              # BaseTruss, softmin, helpers
│   │   └── members.py           # Member classes
│   ├── trusses/                 # Truss implementations (add new trusses here!)
│   │   └── pratt/               # Pratt truss reference implementation
│   │       ├── truss.py         # PrattTruss class
│   │       └── physics.py       # Pratt-specific differentiable physics
│   ├── materials/               # Material library
│   ├── optimization/
│   │   ├── optimizer.py         # BridgeOptimizer
│   │   └── result.py            # OptimizationResult
│   ├── analysis/                # Failure reporting
│   └── utils/                   # Unit conversions
├── examples/                    # Example scripts
├── tests/                       # Test suite
└── docs/                        # Documentation
```

### Adding New Truss Types

To add a new truss type (e.g., Warren truss), create a new subpackage:

```
trusses/
├── pratt/                       # Existing reference implementation
│   ├── __init__.py
│   ├── truss.py
│   └── physics.py
└── warren/                      # Your new truss type
    ├── __init__.py
    ├── truss.py                 # WarrenTruss(BaseTruss)
    └── physics.py               # Warren-specific differentiable physics
```

## 🧪 Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## 🤔 Why Gradient-Based?

Traditional truss optimization uses:
- **Trial and error** — Slow, misses optima
- **Genetic algorithms** — A pain
- **Grid search** — Exponential in parameters

**Gradient-based optimization**:
- Computes exact gradients via autodiff
- Converges in hundreds (not millions) of iterations
- Handles continuous parameters naturally
- The `softmin` trick gives gradients through `min()` operations
- Is cooler

The key insight: structural physics (geometry → forces → stresses → failure) is just math, and math is differentiable!

## 📜 License

MIT License - see [LICENSE](LICENSE)

---

<p align="center">
  Built with PyTorch 🔥 for structural engineering
</p>
