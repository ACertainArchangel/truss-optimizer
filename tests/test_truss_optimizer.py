"""
Test suite for Truss Optimizer.

Run with: pytest tests/ -v
"""

import math
import pytest
import torch

from truss_optimizer import (
    PrattTruss, 
    BridgeOptimizer, 
    materials,
    BaseTruss,
    FailureMode,
    softmin,
    euler_buckling_load,
    DTYPE,
)
from truss_optimizer.core.members import TensionMember, CompressionMember, MemberGeometry
from truss_optimizer.optimization.torch_physics import compute_max_load, compute_volume
from truss_optimizer.utils import inches_to_meters


class TestHelperFunctions:
    """Test helper functions for building custom trusses."""
    
    def test_softmin_approximates_min(self):
        """Softmin should approximate true minimum."""
        values = torch.tensor([100.0, 150.0, 80.0, 200.0], dtype=DTYPE)
        result = softmin(values, alpha=100.0)
        
        # Should be close to true min (80)
        assert abs(result.item() - 80.0) < 1.0
    
    def test_softmin_has_gradients(self):
        """Softmin should provide gradients to all inputs."""
        values = torch.tensor([100.0, 150.0, 80.0], dtype=DTYPE, requires_grad=True)
        result = softmin(values, alpha=100.0)
        result.backward()
        
        # All elements should have non-zero gradients
        assert values.grad is not None
        assert all(g != 0 for g in values.grad)
    
    def test_euler_buckling_formula(self):
        """Euler buckling should follow standard formula."""
        E = torch.tensor(200e9, dtype=DTYPE)  # Steel
        I = torch.tensor(1e-8, dtype=DTYPE)   # m^4
        K = 1.0
        L = torch.tensor(1.0, dtype=DTYPE)    # m
        
        F_cr = euler_buckling_load(E, I, K, L)
        
        # F_cr = π²EI/(KL)²
        expected = (math.pi**2 * 200e9 * 1e-8) / (1.0 * 1.0)**2
        assert abs(F_cr.item() - expected) / expected < 1e-6


class TestMaterials:
    """Test material definitions."""
    
    def test_balsa_wood_properties(self):
        """Balsa wood should have realistic properties."""
        balsa = materials.BalsaWood()
        
        assert 2e9 < balsa.E < 6e9  # Young's modulus reasonable
        assert 5e6 < balsa.sigma_compression < 20e6  # Compression strength
        assert 10e6 < balsa.sigma_tension < 30e6  # Tension strength
        assert 100 < balsa.density < 400  # Density (kg/m³)
    
    def test_custom_material(self):
        """Custom material should accept user-defined properties."""
        custom = materials.Custom(
            E=100e9,
            sigma_compression=50e6,
            sigma_tension=60e6,
            density=1000,
            name="Test Material"
        )
        
        assert custom.E == 100e9
        assert custom.sigma_compression == 50e6
        assert custom.sigma_tension == 60e6
        assert custom.density == 1000
        assert custom.name == "Test Material"


class TestMembers:
    """Test structural member classes."""
    
    def test_tension_member_properties(self):
        """Tension member should compute properties correctly."""
        geom = MemberGeometry(length=1.0, thickness=0.01, depth=0.02)
        member = TensionMember(sigma_tension=20e6, geometry=geom)
        
        assert member.geometry.area == 0.01 * 0.02
        assert member.geometry.volume == 0.01 * 0.02 * 1.0
        assert member.max_axial_load() == 20e6 * 0.01 * 0.02
    
    def test_compression_member_buckling(self):
        """Compression member should compute Euler buckling correctly."""
        geom = MemberGeometry(length=1.0, thickness=0.01, depth=0.02)
        member = CompressionMember(
            sigma_compression=20e6,
            E=200e9,
            K=1.0,
            geometry=geom
        )
        
        # Euler buckling: F_cr = π²EI/(KL)²
        I = geom.I_in_plane
        expected = (math.pi**2 * 200e9 * I) / (1.0 * 1.0)**2
        actual = member.euler_buckling_load(in_plane=True)
        
        assert abs(actual - expected) / expected < 1e-6


class TestPrattTruss:
    """Test Pratt truss implementation."""
    
    @pytest.fixture
    def standard_bridge(self):
        """Create a standard test bridge."""
        return PrattTruss(
            span=inches_to_meters(18.5),
            height=inches_to_meters(6.0),
            angle=math.radians(30),
            material=materials.BalsaWood()
        )
    
    def test_member_creation(self, standard_bridge):
        """Bridge should create all required members."""
        expected_members = {
            'incline', 'diagonal', 'top_chord', 'bottom_chord',
            'mid_vert', 'side_vert'
        }
        assert set(standard_bridge.members.keys()) == expected_members
    
    def test_failure_modes_count(self, standard_bridge):
        """Bridge should have 14 failure modes."""
        failure_modes = standard_bridge.get_failure_modes()
        # 2 tension modes + 12 compression modes = 14
        assert len(failure_modes) == 14
    
    def test_critical_load_positive(self, standard_bridge):
        """Critical load should be positive and finite."""
        assert standard_bridge.critical_load > 0
        assert math.isfinite(standard_bridge.critical_load)
    
    def test_volume_positive(self, standard_bridge):
        """Volume should be positive."""
        assert standard_bridge.volume > 0
    
    def test_weight_positive(self, standard_bridge):
        """Weight should be positive."""
        assert standard_bridge.weight > 0
    
    def test_load_to_weight_positive(self, standard_bridge):
        """Load-to-weight ratio should be positive."""
        assert standard_bridge.load_to_weight > 0


class TestTorchPhysics:
    """Test PyTorch physics implementation."""
    
    @pytest.fixture
    def params(self):
        """Standard parameters for testing."""
        return dict(
            angle=math.radians(30),
            height=0.15,
            length=0.47,
            incline_thickness=0.01,
            diagonal_thickness=0.01,
            mid_vert_thickness=0.01,
            side_vert_thickness=0.01,
            top_thickness=0.01,
            bottom_thickness=0.01,
            incline_depth=0.02,
            diagonal_depth=0.02,
            mid_vert_depth=0.02,
            side_vert_depth=0.02,
            top_depth=0.02,
            bottom_depth=0.02,
            E=3.5e9,
            sigma_compression=12e6,
            sigma_tension=20e6,
        )
    
    def test_max_load_finite(self, params):
        """Max load should be finite and positive."""
        max_load, _ = compute_max_load(**params)
        assert max_load.item() > 0
        assert math.isfinite(max_load.item())
    
    def test_volume_finite(self, params):
        """Volume should be finite and positive."""
        geom_params = {k: v for k, v in params.items() 
                       if k not in ('E', 'sigma_compression', 'sigma_tension')}
        volume = compute_volume(**geom_params)
        assert volume.item() > 0
        assert math.isfinite(volume.item())
    
    def test_gradient_flow(self, params):
        """Gradients should flow through all parameters."""
        # Make thickness a learnable parameter
        thickness = torch.tensor(0.01, dtype=DTYPE, requires_grad=True)
        
        test_params = dict(params)
        test_params['top_thickness'] = thickness
        
        max_load, _ = compute_max_load(**test_params)
        max_load.backward()
        
        # Gradient should exist and be finite
        assert thickness.grad is not None
        assert math.isfinite(thickness.grad.item())
        assert thickness.grad.item() != 0  # Should be non-zero


class TestOptimizer:
    """Test optimizer functionality."""
    
    def test_optimizer_creation(self):
        """Optimizer should initialize correctly."""
        bridge = PrattTruss(
            span=0.47,
            height=0.15,
            material=materials.BalsaWood()
        )
        
        optimizer = BridgeOptimizer(bridge=bridge)
        
        assert optimizer.objective == 'load_to_weight'
        assert optimizer.material is not None
    
    def test_optimization_improves(self):
        """Optimization should improve the objective."""
        bridge = PrattTruss(
            span=0.47,
            height=0.15,
            material=materials.BalsaWood()
        )
        
        optimizer = BridgeOptimizer(
            bridge=bridge,
            objective='load_to_weight',
            learning_rate=0.01
        )
        
        initial_loss = optimizer._compute_loss().item()
        
        # Run a few iterations
        for i in range(50):
            optimizer._train_step(i)
        
        final_loss = optimizer._compute_loss().item()
        
        # Loss should decrease (since we're minimizing negative ratio)
        assert final_loss <= initial_loss
    
    def test_bounds_respected(self):
        """Parameters should stay within bounds."""
        bridge = PrattTruss(
            span=0.47,
            height=0.15,
            material=materials.BalsaWood()
        )
        
        min_height = 0.10
        max_height = 0.20
        
        optimizer = BridgeOptimizer(
            bridge=bridge,
            constraints={'height': (min_height, max_height)},
            learning_rate=0.1  # Large LR to test bounds
        )
        
        # Run many iterations
        for i in range(100):
            optimizer._train_step(i)
        
        height = optimizer.trainable_params['height'].item()
        assert min_height <= height <= max_height


class TestIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test complete optimization workflow."""
        # Create bridge
        bridge = PrattTruss(
            span=inches_to_meters(18.5),
            height=inches_to_meters(6.0),
            angle=math.radians(30),
            material=materials.BalsaWood()
        )
        
        # Create optimizer
        optimizer = BridgeOptimizer(
            bridge=bridge,
            objective='load_to_weight',
            constraints={
                'span': (inches_to_meters(18.5), inches_to_meters(18.5)),
            }
        )
        
        # Run optimization
        result = optimizer.optimize(iterations=100, verbose=False)
        
        # Check result
        assert result.critical_load > 0
        assert result.weight > 0
        assert result.load_to_weight > 0
        assert result.governing_mode in result.failure_modes
        assert len(result.history['loss']) == 100


class TestUnitConversions:
    """Test unit conversion utilities."""
    
    def test_inches_to_meters(self):
        """Inch to meter conversion should be accurate."""
        assert abs(inches_to_meters(1.0) - 0.0254) < 1e-10
        assert abs(inches_to_meters(12.0) - 0.3048) < 1e-10  # 1 foot
    
    def test_roundtrip_conversion(self):
        """Roundtrip conversion should preserve value."""
        from truss_optimizer.utils import meters_to_inches
        
        original = 0.5  # meters
        converted = inches_to_meters(meters_to_inches(original))
        assert abs(converted - original) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
