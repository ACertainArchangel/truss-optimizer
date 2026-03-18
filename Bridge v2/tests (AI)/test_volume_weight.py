"""Tests for volume and weight computations in ``pratt_torch``.

Split into two test cases:

- ``TestVolumeAndWeightCalculations`` — correctness and differentiability of
  ``compute_volume_torch`` and ``compute_weight_torch``.
- A parity check (inside ``TestVolumeAndWeightCalculations``) that verifies
  the torch volume matches ``OnePanelPratt2D.get_total_volume()``.
"""

import math
import unittest

import torch
from bridges_parametric import OnePanelPratt2D
from pratt_torch import DTYPE, compute_volume_torch, compute_weight_torch


class TestVolumeAndWeightCalculations(unittest.TestCase):
    """Test suite for volume and weight calculations with differentiability."""

    def setUp(self):
        """Set up standard parameters."""

        self.params = {
            "incline_thickness": 0.02,
            "diagonal_thickness": 0.02,
            "mid_vert_thickness": 0.02,
            "side_vert_thickness": 0.02,
            "top_thickness": 0.02,
            "bottom_thickness": 0.02,
            "incline_depth": 0.05,
            "diagonal_depth": 0.05,
            "mid_vert_depth": 0.05,
            "side_vert_depth": 0.05,
            "top_depth": 0.05,
            "bottom_depth": 0.05,
            "length": 10.0,
            "height": 2.0,
            "angle": math.radians(30),
        }

    def test_volume_computation(self):
        """Test that volume computation produces finite positive values."""

        volume = compute_volume_torch(
            **self.params,
            dtype=DTYPE,
            device=None,
        )

        self.assertTrue(torch.isfinite(volume), f"Volume should be finite, got {volume}")

        self.assertGreater(volume.item(), 0, f"Volume should be positive, got {volume}")

    def test_weight_computation(self):
        """Test that weight computation produces correct values."""

        volume = compute_volume_torch(**self.params, dtype=DTYPE, device=None)

        weight = compute_weight_torch(**self.params, density=7850.0, dtype=DTYPE, device=None)

        expected_weight = volume * 7850.0 * 9.81

        self.assertAlmostEqual(
            weight.item(),
            expected_weight.item(),
            places=5,
            msg=f"Weight {weight.item()} != expected {expected_weight.item()}",
        )

    def test_weight_with_different_densities(self):
        """Test weight scales linearly with density."""

        weight_steel = compute_weight_torch(**self.params, density=7850.0, dtype=DTYPE, device=None)

        weight_aluminum = compute_weight_torch(
            **self.params, density=2700.0, dtype=DTYPE, device=None
        )

        ratio = weight_steel.item() / weight_aluminum.item()

        expected_ratio = 7850.0 / 2700.0

        self.assertAlmostEqual(
            ratio,
            expected_ratio,
            places=5,
            msg=f"Weight ratio {ratio} != expected {expected_ratio}",
        )

    def test_volume_differentiability_thickness(self):
        """Test that volume is differentiable w.r.t. thickness parameters."""

        params = {k: v for k, v in self.params.items()}

        thickness = torch.tensor(0.02, dtype=DTYPE, requires_grad=True)

        params["top_thickness"] = thickness

        volume = compute_volume_torch(**params, dtype=DTYPE, device=None)

        volume.backward()

        self.assertIsNotNone(thickness.grad, "Gradient should be computed")

        self.assertTrue(torch.isfinite(thickness.grad), "Gradient should be finite")

        self.assertGreater(
            thickness.grad.item(), 0, "Gradient should be positive (more thickness = more volume)"
        )

    def test_volume_differentiability_depth(self):
        """Test that volume is differentiable w.r.t. depth parameters."""

        params = {k: v for k, v in self.params.items()}

        depth = torch.tensor(0.05, dtype=DTYPE, requires_grad=True)

        params["top_depth"] = depth

        volume = compute_volume_torch(**params, dtype=DTYPE, device=None)

        volume.backward()

        self.assertIsNotNone(depth.grad, "Gradient should be computed")

        self.assertTrue(torch.isfinite(depth.grad), "Gradient should be finite")

        self.assertGreater(depth.grad.item(), 0, "Gradient should be positive")

    def test_volume_differentiability_geometry(self):
        """Test that volume is differentiable w.r.t. geometric parameters."""

        params = {k: v for k, v in self.params.items()}

        height = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)

        params["height"] = height

        volume = compute_volume_torch(**params, dtype=DTYPE, device=None)

        volume.backward()

        self.assertIsNotNone(height.grad, "Gradient should be computed")

        self.assertTrue(torch.isfinite(height.grad), "Gradient should be finite")

    def test_weight_differentiability_thickness(self):
        """Test that weight is differentiable w.r.t. thickness."""

        params = {k: v for k, v in self.params.items()}

        thickness = torch.tensor(0.02, dtype=DTYPE, requires_grad=True)

        params["incline_thickness"] = thickness

        weight = compute_weight_torch(**params, density=7850.0, dtype=DTYPE, device=None)

        weight.backward()

        self.assertIsNotNone(thickness.grad, "Gradient should be computed")

        self.assertTrue(torch.isfinite(thickness.grad), "Gradient should be finite")

        self.assertGreater(thickness.grad.item(), 0, "More thickness = more weight")

    def test_weight_differentiability_multiple_params(self):
        """Test that weight supports gradients w.r.t. multiple parameters."""

        params = {k: v for k, v in self.params.items()}

        thickness = torch.tensor(0.02, dtype=DTYPE, requires_grad=True)

        depth = torch.tensor(0.05, dtype=DTYPE, requires_grad=True)

        params["top_thickness"] = thickness

        params["top_depth"] = depth

        weight = compute_weight_torch(**params, density=7850.0, dtype=DTYPE, device=None)

        weight.backward()

        self.assertIsNotNone(thickness.grad, "Thickness gradient should exist")

        self.assertIsNotNone(depth.grad, "Depth gradient should exist")

        self.assertTrue(torch.isfinite(thickness.grad), "Gradients should be finite")

        self.assertTrue(torch.isfinite(depth.grad), "Gradients should be finite")

    def test_parity_with_parametric(self):
        """Test that torch volume matches the parametric version (approximately)."""

        bridge = OnePanelPratt2D(
            angle=self.params["angle"],
            height=self.params["height"],
            length=self.params["length"],
            incline_thickness=self.params["incline_thickness"],
            diagonal_thickness=self.params["diagonal_thickness"],
            mid_vert_thickness=self.params["mid_vert_thickness"],
            side_vert_thickness=self.params["side_vert_thickness"],
            top_thickness=self.params["top_thickness"],
            bottom_thickness=self.params["bottom_thickness"],
            incline_depth=self.params["incline_depth"],
            diagonal_depth=self.params["diagonal_depth"],
            mid_vert_depth=self.params["mid_vert_depth"],
            side_vert_depth=self.params["side_vert_depth"],
            top_depth=self.params["top_depth"],
            bottom_depth=self.params["bottom_depth"],
            E=200e9,
            sigma_compression=250e6,
            sigma_tension=400e6,
        )

        param_volume = bridge.get_total_volume()

        torch_volume = compute_volume_torch(**self.params, dtype=DTYPE, device=None).item()

        rel_error = abs(param_volume - torch_volume) / param_volume

        self.assertLess(
            rel_error,
            0.01,
            msg=f"Torch volume {torch_volume} differs from parametric {param_volume} by {rel_error*100:.2f}%",
        )

    def test_weight_gradient_chain_rule(self):
        """Test that gradient w.r.t. weight scales correctly through chain rule."""

        params = {k: v for k, v in self.params.items()}

        thickness = torch.tensor(0.02, dtype=DTYPE, requires_grad=True)

        params["top_thickness"] = thickness

        weight = compute_weight_torch(**params, density=7850.0, dtype=DTYPE, device=None)

        weight.backward()

        grad_weight_wrt_thickness = thickness.grad.item()

        self.assertGreater(grad_weight_wrt_thickness, 0)

        self.assertLess(grad_weight_wrt_thickness, 1e9, msg="Gradient magnitude seems too large")


if __name__ == "__main__":

    unittest.main()
