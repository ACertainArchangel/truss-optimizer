# Theory: Structural Engineering Background

This document explains the structural engineering principles used in the Truss Optimizer framework. Understanding these concepts will help you implement custom truss types correctly.

## Table of Contents

1. [Truss Fundamentals](#truss-fundamentals)
2. [Failure Modes](#failure-modes)
3. [Euler Buckling](#euler-buckling)
4. [Combined Stress](#combined-stress)
5. [Gradient-Based Optimization](#gradient-based-optimization)
6. [Making Physics Differentiable](#making-physics-differentiable)

## Truss Fundamentals

### What is a Truss?

A truss is a structural framework consisting of straight members connected at joints (nodes). The key characteristics are:

1. **Members carry only axial loads** (tension or compression)
2. **Joints are assumed to be pin connections** (no moment transfer)
3. **Loads are applied only at joints**

This simplifies analysis significantly compared to general frame structures.

### The Pratt Truss

The Pratt truss, patented in 1844 by Thomas and Caleb Pratt, is one of the most common bridge configurations. Its defining feature is:

- **Diagonal members slope toward the center**
- Under typical loading (gravity loads on bottom chord):
  - Diagonals are in **tension**
  - Vertical members are in **compression**

This is advantageous because tension members can be slender (no buckling risk), while compression members (verticals) are shorter than diagonals, reducing buckling concerns.

```
Pratt Truss under load:

    ────────────────────────────────────   (compression)
   /│                                   │\
  ⤢ │                                   │ ⤡  (compression)
 /  ↓                                   ↓  \
/   ↓                                   ↓   \
    ↓         ⤡       ⤢         ⤡       ↓      (diagonals in tension)
────┴─────────────────────────────────────┴────  (tension)
            ↑               ↑
          supports
```

### Method of Joints

To find internal forces, we use the **method of joints**:

At each joint, apply equilibrium:
- ΣFx = 0 (horizontal forces)
- ΣFy = 0 (vertical forces)

For our single-panel Pratt truss with central load F:

**Incline member:**
$$F_{incline} = \frac{F}{2 \sin(\theta)}$$ (compression)

**Diagonal member:**
$$F_{diagonal} = \frac{F}{6 \cos(\phi)}$$ (tension)

**Top/Bottom chord:**
$$F_{chord} = F_{incline} \cos(\theta) + F_{diagonal} \sin(\phi)$$

Where:
- θ = incline angle from horizontal
- φ = diagonal angle from horizontal

## Failure Modes

### Overview

Each structural member can fail in multiple ways. The bridge fails when ANY member fails, so we must check all failure modes and find the minimum critical load.

| Member Type | Failure Modes |
|------------|---------------|
| Tension | Rupture (material yields in tension) |
| Compression | Euler buckling, Material crushing |

### Tension Failure

A tension member fails when the stress exceeds the material's tensile strength:

$$\sigma = \frac{F}{A} \leq \sigma_{tension}$$

With bending moment (combined loading):

$$\sigma = \frac{F}{A} + \frac{M \cdot c}{I} \leq \sigma_{tension}$$

Where:
- F = axial tensile force
- A = cross-sectional area
- M = bending moment
- c = distance to extreme fiber (thickness/2)
- I = second moment of area

### Compression Failure

Compression members can fail by:

1. **Euler buckling** (elastic instability)
2. **Material crushing** (stress exceeds strength)

The critical load is the **minimum** of these two.

## Euler Buckling

### Theory

When a slender column is compressed, it can suddenly deflect sideways (buckle) even though the material hasn't yielded. Euler's formula gives the critical buckling load:

$$F_{cr} = \frac{\pi^2 E I}{(K L)^2}$$

Where:
- E = Young's modulus (material stiffness)
- I = second moment of area (geometric stiffness)
- K = effective length factor
- L = member length

### Effective Length Factor (K)

The K factor accounts for end conditions:

| End Conditions | K | Description |
|---------------|---|-------------|
| Pinned-Pinned | 1.0 | Both ends free to rotate |
| Fixed-Pinned | 0.7 | One end fixed, one pinned |
| Fixed-Fixed | 0.5 | Both ends fixed |
| Fixed-Free | 2.0 | Cantilever column |

In our truss model, we typically use **K = 0.5** (fixed-fixed) because the connections provide rotational restraint.

### In-Plane vs Out-of-Plane Buckling

Members can buckle in two directions:

**In-plane (about weak axis):**
$$I_{in} = \frac{d \cdot t^3}{12}$$

**Out-of-plane (about strong axis):**
$$I_{out} = \frac{t \cdot d^3}{12}$$

Where:
- d = depth (perpendicular to plane)
- t = thickness (in plane)

The member buckles about whichever axis has lower stiffness, so we check both.

### Moment Interaction

When a member has both axial load and bending moment, the interaction formula is:

$$\frac{F_{axial}}{F_{cr}} + \frac{M \cdot c \cdot A}{F_{cr} \cdot I} \leq 1$$

This reduces the buckling capacity when moments are present.

## Combined Stress

### Material Crushing

Even if a member doesn't buckle, it can fail when the combined stress exceeds the material's compressive strength:

$$\sigma_{combined} = \frac{F}{A} + \frac{M \cdot c}{I} \leq \sigma_{compression}$$

### Critical Load Calculation

To find the critical applied load F that causes failure:

1. Express member force as function of F: $F_{member} = a \cdot F$
2. Express member moment as function of F: $M_{member} = b \cdot F$
3. Set up failure equation and solve for F:

$$\sigma_{failure} = \frac{a \cdot F}{A} + \frac{b \cdot F \cdot c}{I} = \sigma_{allowable}$$

$$F_{critical} = \frac{\sigma_{allowable}}{\frac{a}{A} + \frac{b \cdot c}{I}}$$

## Gradient-Based Optimization

### Why Gradient-Based?

Traditional structural optimization uses:
- Grid search (slow, curse of dimensionality)
- Genetic algorithms (slow, no convergence guarantee)
- Manual iteration (expertise required)

Gradient-based optimization is:
- **Fast** - uses derivative information
- **Efficient** - scales to many parameters
- **Converges** - to local optima

### Automatic Differentiation

PyTorch provides **automatic differentiation** (autodiff):

```python
import torch

# All calculations are tracked
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x

# Compute gradient automatically
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
```

This means we can:
1. Write physics equations naturally
2. PyTorch computes all gradients automatically
3. Optimizer uses gradients to improve parameters

### Differentiable Minimum (Softmin)

The critical load is:

$$F_{critical} = \min(F_1, F_2, ..., F_{14})$$

But `min()` isn't differentiable (gradient is zero almost everywhere).

We use **softmin** - a smooth approximation:

$$\text{softmin}(\mathbf{F}) = -T \cdot \log\sum_i e^{-F_i/T}$$

Where T (temperature) controls smoothness:
- T → 0: approaches true min
- T large: smoother approximation

### Optimization Algorithm

We use **Adam** optimizer:

1. Initialize parameters randomly within bounds
2. Compute loss (negative load-to-weight ratio)
3. Compute gradients via autodiff
4. Update parameters using Adam rule
5. Clamp to bounds
6. Repeat until convergence

```python
for iteration in range(5000):
    loss = compute_loss(params)    # Forward pass
    loss.backward()                 # Backward pass (compute gradients)
    optimizer.step()                # Update parameters
    apply_bounds(params)            # Enforce constraints
```

### Objective Functions

| Objective | Loss Function | Goal |
|-----------|--------------|------|
| max_load | $-F_{critical}$ | Maximize load |
| min_weight | $W$ | Minimize weight |
| load_to_weight | $-\frac{F_{critical}}{W}$ | Maximize ratio |

### Local vs Global Optima

Gradient descent finds **local** optima. To find better solutions:

1. **Multiple random starts** - run many trials with different seeds
2. **Learning rate tuning** - try different step sizes
3. **Constraint relaxation** - start with loose bounds, tighten gradually

Our `run_optimization_trials()` function automates multi-start optimization.

## Summary

The Truss Optimizer framework combines:

1. **Structural mechanics** - accurate physics modeling of failure modes
2. **Automatic differentiation** - efficient gradient computation via PyTorch
3. **Gradient descent** - fast convergence to optimal designs

When creating custom truss types, the key is ensuring your physics are:
- **Complete** - model all relevant failure modes
- **Differentiable** - use torch operations throughout
- **Correctly connected** - failure loads scale properly with applied load

## Making Physics Differentiable

### The Key Challenge

Standard structural analysis code isn't differentiable. Consider:

```python
# Non-differentiable - can't optimize this!
critical_load = min(F_tension, F_buckling, F_crush)
```

The `min()` function has zero gradient for non-minimum elements. The optimizer can't "see" that other failure modes exist or how close they are.

### The softmin Solution

We use a **smooth approximation** of minimum:

$$\text{softmin}(\mathbf{F}) = -T \cdot \log\sum_i e^{-F_i/T}$$

This gives:
- Approximate minimum value
- Non-zero gradients for all elements
- Gradients proportional to how close each value is to the minimum

```python
from truss_optimizer import softmin

# Differentiable - optimizer can learn from all modes!
failure_loads = torch.stack([F_tension, F_buckling, F_crush])
critical_load = softmin(failure_loads, alpha=100.0)
```

### Other Differentiability Tips

1. **Avoid conditionals on tensors**:
   ```python
   # Bad - not differentiable
   if load > capacity:
       return 0
   
   # Good - use torch.where or torch.clamp
   result = torch.clamp(capacity - load, min=0)
   ```

2. **Use torch math functions**:
   ```python
   # Good
   import torch
   length = torch.sqrt(dx**2 + dy**2)
   ```

3. **Convert inputs properly**:
   ```python
   # Allows both float and tensor inputs
   x = torch.as_tensor(x, dtype=DTYPE)
   ```

## References

1. Timoshenko, S.P. & Gere, J.M. (1961). *Theory of Elastic Stability*. McGraw-Hill.
2. Hibbeler, R.C. (2011). *Structural Analysis*. Prentice Hall.
3. Paszke, A. et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*.
