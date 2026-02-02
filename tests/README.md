# Tests

This directory contains the test suite for Truss Optimizer.

## Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=truss_optimizer --cov-report=html

# Run specific test class
pytest tests/test_truss_optimizer.py::TestMaterials -v

# Run specific test
pytest tests/test_truss_optimizer.py::TestTorchPhysics::test_gradient_flow -v
```

## Test Structure

- `test_truss_optimizer.py` - Main test file covering:
  - `TestMaterials` - Material property tests
  - `TestMembers` - Structural member calculations
  - `TestPrattTruss` - Truss model tests
  - `TestTorchPhysics` - PyTorch implementation tests
  - `TestOptimizer` - Optimization functionality
  - `TestIntegration` - End-to-end workflow tests
  - `TestUnitConversions` - Unit conversion utilities

## Writing Tests

New tests should follow the existing patterns:

```python
class TestNewFeature:
    """Test new feature."""
    
    @pytest.fixture
    def setup_data(self):
        """Create test fixtures."""
        return ...
    
    def test_expected_behavior(self, setup_data):
        """Describe what should happen."""
        result = function_under_test(setup_data)
        assert result == expected
```
