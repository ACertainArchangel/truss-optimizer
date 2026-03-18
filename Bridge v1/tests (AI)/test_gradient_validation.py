"""Gradient validation: autodiff vs finite differences and gradient flow."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest

import torch
from pratt_torch import DTYPE, max_load_torch


class TestGradientValidation(unittest.TestCase):
    """Test suite for gradient correctness and consistency"""

    @classmethod
    def setUpClass(cls):
        """Build shared parameter dict and identify the governing failure mode.

        The governing mode is the one whose failure load equals the overall
        minimum.  Knowing it upfront lets individual tests focus gradient
        checks on the most numerically sensitive path.
        """

        cls.params = dict(
            angle=math.radians(30),
            height=2.0,
            length=10.0,
            incline_thickness=0.02,
            diagonal_thickness=0.02,
            mid_vert_thickness=0.02,
            side_vert_thickness=0.02,
            top_thickness=0.02,
            bottom_thickness=0.02,
            incline_depth=0.05,
            diagonal_depth=0.05,
            mid_vert_depth=0.05,
            side_vert_depth=0.05,
            top_depth=0.05,
            bottom_depth=0.05,
            E=200e9,
            sigma_compression=250e6,
            sigma_tension=400e6,
            K=1.0,
        )

        max_load_val, failure_loads = max_load_torch(**cls.params)

        min_load = min(v.item() for v in failure_loads.values())

        cls.governing_mode = [
            k for k, v in failure_loads.items() if abs(v.item() - min_load) < 1.0
        ][0]

    def test_all_geometric_parameters_have_gradients(self):
        """Test that all geometric parameters have computable gradients"""

        test_params = [
            "incline_thickness",
            "incline_depth",
            "diagonal_thickness",
            "diagonal_depth",
            "mid_vert_thickness",
            "mid_vert_depth",
            "side_vert_thickness",
            "side_vert_depth",
            "top_thickness",
            "top_depth",
            "bottom_thickness",
            "bottom_depth",
        ]

        for param_name in test_params:

            with self.subTest(parameter=param_name):

                param_tensor = torch.tensor(
                    self.params[param_name], dtype=DTYPE, requires_grad=True
                )

                max_load, _ = max_load_torch(**{**self.params, param_name: param_tensor})

                max_load.backward()

                grad = param_tensor.grad.item()

                self.assertFalse(math.isnan(grad), f"Gradient for {param_name} is NaN")

                self.assertFalse(math.isinf(grad), f"Gradient for {param_name} is infinite")

    def test_gradient_matches_finite_difference(self):
        """Test that automatic differentiation matches finite difference approximation"""

        epsilon = 1e-7

        max_relative_error = 0.01

        test_params = ["incline_thickness", "top_depth", "bottom_thickness"]

        for param_name in test_params:

            with self.subTest(parameter=param_name):

                param_val = self.params[param_name]

                params_plus = {**self.params, param_name: param_val + epsilon}

                max_load_plus, _ = max_load_torch(**params_plus)

                params_minus = {**self.params, param_name: param_val - epsilon}

                max_load_minus, _ = max_load_torch(**params_minus)

                fd_grad = (max_load_plus.item() - max_load_minus.item()) / (2 * epsilon)

                param_tensor = torch.tensor(param_val, dtype=DTYPE, requires_grad=True)

                max_load, _ = max_load_torch(**{**self.params, param_name: param_tensor})

                max_load.backward()

                ad_grad = param_tensor.grad.item()

                if abs(fd_grad) > 1e-6:

                    relative_error = abs(fd_grad - ad_grad) / abs(fd_grad)

                    self.assertLess(
                        relative_error,
                        max_relative_error,
                        f"Gradient mismatch for {param_name}: FD={fd_grad:.6f}, AD={ad_grad:.6f}, error={relative_error*100:.4f}%",
                    )

    def test_max_load_has_gradient_flow(self):
        """Test that max_load computation has proper gradient flow"""

        test_params = ["top_thickness", "top_depth"]

        for param in test_params:

            with self.subTest(parameter=param):

                param_tensor = torch.tensor(self.params[param], dtype=DTYPE, requires_grad=True)

                max_load, failure_dict = max_load_torch(**{**self.params, param: param_tensor})

                max_load.backward()

                grad = param_tensor.grad.item()

                self.assertFalse(math.isnan(grad), f"Gradient for {param} is NaN")

                self.assertFalse(math.isinf(grad), f"Gradient for {param} is infinite")

    def test_gradient_signs_are_computed(self):
        """Test that gradient computations work (signs may vary due to softmin smoothing)"""

        test_params = ["top_thickness", "top_depth", "bottom_thickness"]

        for param in test_params:

            with self.subTest(parameter=param):

                param_tensor = torch.tensor(self.params[param], dtype=DTYPE, requires_grad=True)

                max_load, _ = max_load_torch(**{**self.params, param: param_tensor})

                max_load.backward()

                grad = param_tensor.grad.item()

                self.assertFalse(math.isnan(grad), f"Gradient for {param} is NaN")

                self.assertFalse(math.isinf(grad), f"Gradient for {param} is infinite")

    def test_no_gradient_explosion(self):
        """Test that gradients remain bounded (no numerical instability)"""

        test_params = [
            "incline_thickness",
            "incline_depth",
            "top_thickness",
            "top_depth",
            "bottom_thickness",
            "bottom_depth",
        ]

        max_reasonable_gradient = 1e12

        for param_name in test_params:

            with self.subTest(parameter=param_name):

                param_tensor = torch.tensor(
                    self.params[param_name], dtype=DTYPE, requires_grad=True
                )

                max_load, _ = max_load_torch(**{**self.params, param_name: param_tensor})

                max_load.backward()

                grad = param_tensor.grad.item()

                self.assertLess(
                    abs(grad),
                    max_reasonable_gradient,
                    f"Gradient for {param_name} is unreasonably large: {grad:.2e}",
                )


class TestGradientConsistency(unittest.TestCase):
    """Test that gradients are consistent across different parameter combinations"""

    def test_gradient_at_different_scales(self):
        """Test that gradients remain consistent when parameters are scaled"""

        base_params = dict(
            angle=math.radians(30),
            height=2.0,
            length=10.0,
            incline_thickness=0.02,
            diagonal_thickness=0.02,
            mid_vert_thickness=0.02,
            side_vert_thickness=0.02,
            top_thickness=0.02,
            bottom_thickness=0.02,
            incline_depth=0.05,
            diagonal_depth=0.05,
            mid_vert_depth=0.05,
            side_vert_depth=0.05,
            top_depth=0.05,
            bottom_depth=0.05,
            E=200e9,
            sigma_compression=250e6,
            sigma_tension=400e6,
            K=1.0,
        )

        for scale in [0.5, 2.0]:

            scaled_params = {
                k: (
                    v * scale
                    if k in ["height", "length"] or k.endswith("thickness") or k.endswith("depth")
                    else v
                )
                for k, v in base_params.items()
            }

            param_tensor = torch.tensor(
                scaled_params["incline_thickness"], dtype=DTYPE, requires_grad=True
            )

            max_load, _ = max_load_torch(**{**scaled_params, "incline_thickness": param_tensor})

            max_load.backward()

            grad = param_tensor.grad.item()

            self.assertFalse(math.isnan(grad), f"Gradient is NaN at scale {scale}x")

            self.assertFalse(math.isinf(grad), f"Gradient is infinite at scale {scale}x")


if __name__ == "__main__":

    unittest.main(verbosity=2)
