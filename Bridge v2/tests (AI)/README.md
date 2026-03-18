# Test Suite for Deep Learning Bridge Project

This directory contains comprehensive tests for the bridge optimization project, validating both the parametric and PyTorch implementations across all failure modes.

## Test Files

### Analysis Tests (New)
- **`test_analyse_parity.py`** - Verifies pratt_analyse.py and pratt_analyse_v2.py produce identical results
- **`test_analyse_methods_parity.py`** - Ensures analysis scripts match get_failure_mode_dict() and PyTorch
- **`test_weight_display.py`** - Tests weight calculation and display with density parameter

### Core Parity Tests
- **`test_complete_parity.py`** - Main validation suite
  - Compares parametric (`bridges_parametric.py`) vs torch (`pratt_torch.py`) implementations
  - Tests 14 failure modes across 3 scenarios (standard, slender, stocky)
  - Validates 100% agreement between implementations
  - **Status**: ✅ All tests passing (14/14 modes match perfectly)

### Gradient Validation
- **`test_gradient_validation.py`** - Automatic differentiation tests
  - Validates PyTorch gradients match finite difference approximations
  - Tests all geometric parameters for gradient flow
  - Checks numerical stability (no NaN/inf gradients)
  - **Status**: ✅ All 6 test cases passing

### Component Tests
- **`test_bridges_torch_parity.py`** - Comprehensive PyTorch vs parametric bridge tests
- **`test_failure_mode_dict.py`** - Tests failure mode dictionary generation
- **`test_moment_inverses.py`** - Validates moment inversion functions
- **`test_volume_weight.py`** - Tests volume and weight calculations
- **`test_pratt2d_loads.py`** - Tests load distribution
- **`test_bridge_optimizer.py`** - Optimizer functionality tests
- **`test_bridge_optimizer_material.py`** - Material-specific optimizer tests

## Running Tests

### Run All Tests (Recommended)
```bash
python tests/run_all_tests.py
```
This runs all 50 tests and provides a comprehensive summary.

### Run Specific Test Suite
```bash
# Analysis parity tests
python tests/test_analyse_parity.py
python tests/test_analyse_methods_parity.py
python tests/test_weight_display.py

# Complete parity tests (14 failure modes)
python tests/test_complete_parity.py

# Gradient validation
python tests/test_gradient_validation.py
```

### Using unittest discover
```bash
python -m unittest discover tests
```

## Test Coverage

### Failure Modes Validated (14 total)
#### Tension Members (2 modes)
- `diagonal_rupture` - Diagonal member tension failure
- `bottom_chord_rupture` - Bottom chord tension failure with moment interaction

#### Compression Members (12 modes = 4 members × 3 modes each)
For each of: incline, top_chord, mid_vert, side_vert
1. `*_buckle` - Euler buckling in-plane with moment interaction
2. `*_buckle_out_of_plane` - Euler buckling out-of-plane with moment interaction  
3. `*_combined_stress` - Material strength with combined axial + bending stress

### Test Scenarios
1. **Standard Configuration** - Typical bridge dimensions
   - Governing mode: `top_chord_buckle_out_of_plane` (54.06 N)
   - Tests all failure modes in balanced design

2. **Slender Members** - Long, thin members (Euler dominant)
   - Governing mode: `top_chord_buckle_out_of_plane` (0.05 N)
   - Tests buckling-dominated failures

3. **Stocky Members** - Short, thick members (material dominant)
   - Governing mode: `top_chord_buckle_out_of_plane` (57147 N)
   - Tests material strength-dominated failures

## Key Findings & Fixes

### Two-Root Problem (Resolved ✅)
- **Issue**: Tension members with moments had equations with two roots:
  - Positive root: F > 0 (compression/pushing down on bridge)
  - Negative root: F < 0 (tension/pulling up on bridge)
- **Solution**: 
  - Use multiple initial guesses in `fsolve`
  - Only accept positive roots (F > 1e-6)
  - Return minimum positive root (most conservative)

### Moment Sign Convention (Resolved ✅)
- **Issue**: Moment can increase tension on one face, decrease on other
- **Solution**: Use `abs(M)` for conservative worst-case stress calculation
- **Result**: Perfect agreement between parametric and torch implementations

### Gradient Flow
- All 12 geometric parameters have valid gradients
- Gradients match finite difference within 1% error
- Softmin smoothing may cause very small gradients for non-governing modes
- This is expected behavior and enables smooth optimization

## Success Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Parity test pass rate | 100% | 100% (42/42 tests) | ✅ |
| Failure modes matching | 14/14 | 14/14 | ✅ |
| Gradient tests passing | 100% | 100% (6/6 tests) | ✅ |
| FD vs AD error | < 1% | < 0.000001% | ✅ |

## Test History

### Major Milestones
1. ✅ Implemented all 14 failure modes with moments
2. ✅ Added Euler buckling (in-plane and out-of-plane)
3. ✅ Fixed parametric implementation bugs (missing Euler checks)
4. ✅ Fixed two-root problem in tension member calculations
5. ✅ Fixed moment sign convention for conservative stress
6. ✅ Achieved 100% parity across all test scenarios
7. ✅ Validated gradient flow for optimization

### Bugs Fixed
- Parametric `find_buckle_F` was missing Euler buckling entirely
- TensionMember I formula was using wrong dimensions
- Moment signs needed absolute value for conservative design
- `fsolve` finding wrong root due to moment interaction

## Notes
- Parametric implementation now matches torch perfectly
- Both implementations include complete physics
- Torch implementation is fully differentiable for gradient-based optimization
- All critical structural failure modes are captured
