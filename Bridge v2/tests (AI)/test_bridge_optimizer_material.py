"""Integration tests verifying that BridgeOptimizer correctly ingests a
Material dataclass and marks the three material parameters (E,
sigma_compression, sigma_tension) as fixed (non-trainable) tensors.
"""

import math
import unittest
from dataclasses import dataclass

from pratt_optimizer import BridgeDesignParams, BridgeOptimizer


@dataclass
class Material:
    """Thin material descriptor used to pass mechanical properties into the optimizer."""

    E: float

    sigma_compression: float

    sigma_tension: float


class TestBridgeOptimizerMaterialIntegration(unittest.TestCase):
    """Tests that verify material-parameter integration in BridgeOptimizer.

    When a Material object is supplied, the optimizer should:
    - override the default E / sigma_compression / sigma_tension values with
      the material's values, and
    - mark all three as fixed (requires_grad=False) so they are not updated
      during training.
    """

    def test_material_integration_sets_fixed_params(self):
        """Material values should be loaded and frozen in the optimizer."""

        initial = BridgeDesignParams()

        material = Material(E=210e9, sigma_compression=300e6, sigma_tension=450e6)

        optimizer = BridgeOptimizer(initial_params=initial, material=material, learning_rate=0.01)

        self.assertIn("E", optimizer.trainable_params)

        self.assertAlmostEqual(optimizer.trainable_params["E"].item(), 210e9)

        self.assertFalse(optimizer.trainable_params["E"].requires_grad)

        self.assertAlmostEqual(optimizer.trainable_params["sigma_compression"].item(), 300e6)

        self.assertFalse(optimizer.trainable_params["sigma_compression"].requires_grad)

        self.assertAlmostEqual(optimizer.trainable_params["sigma_tension"].item(), 450e6)

        self.assertFalse(optimizer.trainable_params["sigma_tension"].requires_grad)


if __name__ == "__main__":

    unittest.main()
