"""Tests for the gradient-based ``BridgeOptimizer``.

Covers:
- Correct classification of fixed vs trainable parameters.
- Loss function computability and gradient attachment.
- Monotone improvement in critical load after training.
- Parameter bound enforcement.
- Immutability of fixed parameters during training.
- Recording of training history.
- Computation of per-parameter deltas between two snapshots.
"""

import math
import unittest

from pratt_optimizer import BridgeDesignParams, BridgeOptimizer


class TestBridgeOptimizer(unittest.TestCase):
    """Test suite for BridgeOptimizer."""

    def setUp(self):
        """Set up a standard optimizer for testing."""

        self.initial_params = BridgeDesignParams(
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
        )

    def test_optimizer_initialization(self):
        """Test that optimizer initializes with correct fixed/trainable parameters."""

        fixed_params = {"E": True, "sigma_compression": True, "sigma_tension": True}

        optimizer = BridgeOptimizer(
            initial_params=self.initial_params,
            fixed_params=fixed_params,
            learning_rate=0.1,
        )

        self.assertIn("angle", optimizer.trainable_params)

        self.assertIn("height", optimizer.trainable_params)

        self.assertIn("top_thickness", optimizer.trainable_params)

        self.assertFalse(optimizer.trainable_params["E"].requires_grad)

        self.assertFalse(optimizer.trainable_params["sigma_compression"].requires_grad)

        self.assertFalse(optimizer.trainable_params["sigma_tension"].requires_grad)

    def test_loss_function_computable(self):
        """Test that loss function can be computed."""

        optimizer = BridgeOptimizer(
            initial_params=self.initial_params,
            fixed_params={"E": True, "sigma_compression": True, "sigma_tension": True},
        )

        loss = optimizer.loss_function()

        self.assertTrue(loss.requires_grad, "Loss should require gradients")

        self.assertTrue(float("-inf") < loss.item() < float("inf"), "Loss should be finite")

    def test_training_improves_critical_load(self):
        """Test that training increases critical load."""

        optimizer = BridgeOptimizer(
            initial_params=self.initial_params,
            fixed_params={
                "E": True,
                "sigma_compression": True,
                "sigma_tension": True,
                "length": True,
            },
            param_bounds={
                "height": (0.5, 5.0),
                "angle": (math.radians(15), math.radians(60)),
                "top_thickness": (0.005, 0.05),
                "bottom_thickness": (0.005, 0.05),
            },
            learning_rate=0.05,
            ratio_mode=None,
        )

        initial_crit_load = -optimizer.loss_function().item()

        init_params, final_params = optimizer.train(num_iterations=50, verbose=False)

        final_crit_load = optimizer.critical_load_history[-1]

        self.assertGreater(
            final_crit_load,
            initial_crit_load,
            msg=f"Final critical load {final_crit_load} should exceed initial {initial_crit_load}",
        )

    def test_parameter_bounds_respected(self):
        """Test that parameter bounds are enforced."""

        param_bounds = {
            "height": (1.0, 2.0),
            "top_thickness": (0.01, 0.03),
        }

        optimizer = BridgeOptimizer(
            initial_params=self.initial_params,
            fixed_params={"E": True, "sigma_compression": True, "sigma_tension": True},
            param_bounds=param_bounds,
            learning_rate=0.1,
        )

        _, final_params = optimizer.train(num_iterations=100, verbose=False)

        height = final_params["height"]

        top_thick = final_params["top_thickness"]

        self.assertGreaterEqual(height, param_bounds["height"][0])

        self.assertLessEqual(height, param_bounds["height"][1])

        self.assertGreaterEqual(top_thick, param_bounds["top_thickness"][0])

        self.assertLessEqual(top_thick, param_bounds["top_thickness"][1])

    def test_fixed_parameters_unchanged(self):
        """Test that fixed parameters are not modified during training."""

        fixed_params = {
            "E": True,
            "sigma_compression": True,
            "sigma_tension": True,
            "length": True,
        }

        optimizer = BridgeOptimizer(
            initial_params=self.initial_params,
            fixed_params=fixed_params,
        )

        init_params, final_params = optimizer.train(num_iterations=50, verbose=False)

        for fixed_param in ["E", "sigma_compression", "sigma_tension", "length"]:

            self.assertEqual(
                init_params[fixed_param],
                final_params[fixed_param],
                msg=f"{fixed_param} should not change during training",
            )

    def test_training_history_recorded(self):
        """Test that training history is recorded properly."""

        optimizer = BridgeOptimizer(
            initial_params=self.initial_params,
            fixed_params={"E": True, "sigma_compression": True, "sigma_tension": True},
        )

        num_iters = 30

        optimizer.train(num_iterations=num_iters, verbose=False)

        history = optimizer.get_optimization_history()

        self.assertEqual(len(history["loss_history"]), num_iters)

        self.assertEqual(len(history["critical_load_history"]), num_iters)

        self.assertGreater(len(history["param_history"]), 0)

    def test_param_deltas_computation(self):
        """Test that parameter deltas are computed correctly."""

        optimizer = BridgeOptimizer(
            initial_params=self.initial_params,
            fixed_params={"E": True, "sigma_compression": True, "sigma_tension": True},
        )

        init = {"a": 1.0, "b": 2.0}

        final = {"a": 1.5, "b": 2.0}

        deltas = optimizer.get_param_deltas(init, final)

        self.assertAlmostEqual(deltas["a"], 0.5)

        self.assertAlmostEqual(deltas["b"], 0.0)


if __name__ == "__main__":

    unittest.main()
