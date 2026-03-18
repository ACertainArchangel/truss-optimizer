"""
Bridge optimizer: trains bridge parameters to maximize critical load.
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
from config import K_EFFECTIVE_LENGTH
from materials import HighDensityBalsaWood, Material
from pratt_torch import DTYPE, compute_weight_torch, max_load_torch
from utils import feet_to_meters, inches_to_meters


@dataclass
class BridgeDesignParams:
    """Container for bridge design parameters (both fixed and trainable)."""

    material: Optional[Material] = None

    angle: float = math.radians(30)

    height: float = inches_to_meters(6.0)

    length: float = inches_to_meters(18.5)

    incline_thickness: float = 0.02

    diagonal_thickness: float = 0.02

    mid_vert_thickness: float = 0.02

    side_vert_thickness: float = 0.02

    top_thickness: float = 0.02

    bottom_thickness: float = 0.02

    incline_depth: float = 0.05

    diagonal_depth: float = 0.05

    mid_vert_depth: float = 0.05

    side_vert_depth: float = 0.05

    top_depth: float = 0.05

    bottom_depth: float = 0.05

    E: float = 200e9

    sigma_compression: float = 250e6

    sigma_tension: float = 400e6

    density: float = 7850.0

    def __post_init__(self):
        """Override material properties if a Material instance is provided."""

        if self.material is not None:

            self.E = self.material.E

            self.sigma_compression = self.material.sigma_compression

            self.sigma_tension = self.material.sigma_tension

            self.density = self.material.density


def random_initial_params(
    bounds: Dict[str, Tuple[float, float]] = None,
    material: Optional[Material] = None,
    seed: Optional[int] = None,
) -> BridgeDesignParams:
    """Generate random initial bridge design parameters within given bounds."""

    import random

    if seed is not None:

        random.seed(seed)

    params = {}

    for field in BridgeDesignParams.__dataclass_fields__.values():

        if field.name == "height":

            max_height = (
                params.get("length", inches_to_meters(18.5)) * math.tan(params.get("angle"))
            ) / 2

            if bounds and "height" in bounds:

                min_val, max_val = bounds["height"]

                max_val = min(max_val, max_height)

                if min_val >= max_val:

                    print(
                        f"It seems {math.degrees(params.get("angle"))} degrees is an impossible angle given the minimum height provided for the competition constraints."
                    )

                    raise ValueError(
                        f"Invalid bounds for height: min {min_val} >= max {max_val} (max allowed by geometry is {max_height})"
                    )

                params["height"] = random.uniform(min_val, max_val)

            else:

                params["height"] = random.uniform(inches_to_meters(4.0), max_height)

            continue

        name = field.name

        if bounds and name in bounds:

            min_val, max_val = bounds[name]

            params[name] = random.uniform(min_val, max_val)

        else:

            params[name] = field.default

    if material is not None:

        params["material"] = material

    return BridgeDesignParams(**params)


class BridgeOptimizer:
    """
    Optimizer for bridge design parameters using PyTorch autodiff.
    """

    def __init__(
        self,
        initial_params: BridgeDesignParams,
        fixed_params: Dict[str, bool] = None,
        material: Optional[object] = None,
        param_bounds: Dict[str, Tuple[float, float]] = None,
        learning_rate: float = 0.1,
        device: Optional[str] = None,  # CPU or CUDA device
        ratio_mode: Optional[str] = "max_load_per_weight",
        fixed_cost=0.0,  # Newtons
        verbose: bool = True,
    ):
        """Initialize optimizer with design params, bounds, and training config."""

        self.device = device or "cpu"

        self.learning_rate = learning_rate

        self.initial_params = initial_params

        self.initial_params_dict = asdict(initial_params)

        if material is not None:

            mat_vals = {}

            for attr in ("E", "sigma_compression", "sigma_tension"):

                if hasattr(material, attr):

                    mat_vals[attr] = getattr(material, attr)

            temp_init = dict(self.initial_params_dict)

            temp_init.update(mat_vals)

            self.initial_params_dict = temp_init

        if fixed_params is None:

            fixed_params = {
                "E": True,
                "sigma_compression": True,
                "sigma_tension": True,
            }

        self.fixed_params = (
            fixed_params
            if fixed_params != "default"
            else {
                "E": True,
                "sigma_compression": True,
                "sigma_tension": True,
                "density": True,
                "length": True,
                "material": True,
            }
        )

        self.param_bounds = param_bounds or {}

        self.trainable_params = {}

        for param_name, param_value in self.initial_params_dict.items():

            is_fixed = self.fixed_params.get(param_name, False)

            requires_grad = not is_fixed

            if isinstance(param_value, (float, int)):

                tensor = torch.tensor(
                    param_value, dtype=DTYPE, device=self.device, requires_grad=requires_grad
                )

                self.trainable_params[param_name] = tensor

        trainable_tensors = [
            v for k, v in self.trainable_params.items() if not self.fixed_params.get(k, False)
        ]

        self.optimizer = torch.optim.Adam(trainable_tensors, lr=learning_rate)

        self.apply_bounds()

        self.loss_history = []

        self.critical_load_history = []

        self.param_history = {name: [] for name in self.trainable_params}

        self.best_objective_value = float("inf")

        self.best_params: Dict[str, float] = {}

        self.best_iteration = -1

        assert (ratio_mode is None) or (
            ratio_mode in ("weight_per_load", "max_load_per_weight")
        ), "invalid ratio_mode"

        self.ratio_mode = ratio_mode

        self.density = initial_params.density

        self.fixed_cost = fixed_cost

        self.verbose = verbose

        if fixed_cost != 0.0 and verbose:

            print(
                f"ℹ  Epoxy weight per plane: {fixed_cost:.4f} N (included in accurate weight calculation with lateral supports)"
            )

        self._last_summary_time = 0.0

    def loss_function(self) -> torch.Tensor:
        """Compute loss based on ratio_mode: None=-load, weight_per_load, or max_load_per_weight."""

        max_load, _ = max_load_torch(
            angle=self.trainable_params["angle"],
            height=self.trainable_params["height"],
            length=self.trainable_params["length"],
            incline_thickness=self.trainable_params["incline_thickness"],
            diagonal_thickness=self.trainable_params["diagonal_thickness"],
            mid_vert_thickness=self.trainable_params["mid_vert_thickness"],
            side_vert_thickness=self.trainable_params["side_vert_thickness"],
            top_thickness=self.trainable_params["top_thickness"],
            bottom_thickness=self.trainable_params["bottom_thickness"],
            incline_depth=self.trainable_params["incline_depth"],
            diagonal_depth=self.trainable_params["diagonal_depth"],
            mid_vert_depth=self.trainable_params["mid_vert_depth"],
            side_vert_depth=self.trainable_params["side_vert_depth"],
            top_depth=self.trainable_params["top_depth"],
            bottom_depth=self.trainable_params["bottom_depth"],
            E=self.trainable_params["E"],
            sigma_compression=self.trainable_params["sigma_compression"],
            sigma_tension=self.trainable_params["sigma_tension"],
            K=K_EFFECTIVE_LENGTH,
            softness=1e-6,
            dtype=DTYPE,
            device=self.device,
        )

        if self.ratio_mode is None:

            return -max_load

        from pratt_torch import compute_weight_with_laterals_torch

        weight = compute_weight_with_laterals_torch(
            incline_thickness=self.trainable_params["incline_thickness"],
            diagonal_thickness=self.trainable_params["diagonal_thickness"],
            mid_vert_thickness=self.trainable_params["mid_vert_thickness"],
            side_vert_thickness=self.trainable_params["side_vert_thickness"],
            top_thickness=self.trainable_params["top_thickness"],
            bottom_thickness=self.trainable_params["bottom_thickness"],
            incline_depth=self.trainable_params["incline_depth"],
            diagonal_depth=self.trainable_params["diagonal_depth"],
            mid_vert_depth=self.trainable_params["mid_vert_depth"],
            side_vert_depth=self.trainable_params["side_vert_depth"],
            top_depth=self.trainable_params["top_depth"],
            bottom_depth=self.trainable_params["bottom_depth"],
            length=self.trainable_params["length"],
            height=self.trainable_params["height"],
            angle=self.trainable_params["angle"],
            density=self.density,
            epoxy_cost=self.fixed_cost,
            dtype=DTYPE,
            device=self.device,
        )

        max_load_clamped = torch.clamp(max_load, min=1e-6)

        weight_clamped = torch.clamp(weight, min=1e-12)

        if self.ratio_mode == "weight_per_load":

            return weight_clamped / max_load_clamped

        else:

            return -max_load_clamped / weight_clamped

    def apply_bounds(self):
        """Clamp trainable parameters to their bounds (but not fixed ones)."""

        for param_name, (min_val, max_val) in self.param_bounds.items():

            if param_name in self.trainable_params and not self.fixed_params.get(param_name, False):

                with torch.no_grad():

                    self.trainable_params[param_name].clamp_(min_val, max_val)

    def train_step(self, iteration: int) -> float:
        """Single training step. Returns scalar loss value."""

        self.optimizer.zero_grad()

        loss = self.loss_function()

        loss_val = loss.item()

        if loss_val < self.best_objective_value:

            self.best_objective_value = loss_val

            self.best_iteration = iteration

            self.best_params = {}

            for name, tensor in self.trainable_params.items():

                self.best_params[name] = tensor.detach().item()

            for name, value in self.initial_params_dict.items():

                if name not in self.best_params:

                    self.best_params[name] = value

        loss.backward()

        self.optimizer.step()

        self.apply_bounds()

        if self.ratio_mode is None:

            crit_load_val = -loss_val

        else:

            crit_load_val = loss_val

        self.loss_history.append(loss_val)

        self.critical_load_history.append(crit_load_val)

        for param_name, param_tensor in self.trainable_params.items():

            self.param_history[param_name].append(param_tensor.item())

        return loss_val

    def train(
        self, num_iterations: int, verbose: bool = True, log_interval: int = 10
    ) -> Tuple[Dict, Dict]:
        """Train for num_iterations steps. Returns (initial_params_dict, best_params_dict)."""

        for iteration in range(num_iterations):

            loss = self.train_step(iteration)

            if verbose and (iteration + 1) % log_interval == 0:

                if self.ratio_mode is None:

                    crit_load = -loss

                    print(
                        f"[Iter {iteration + 1:4d}] Loss: {loss:10.2f} | Critical Load: {crit_load:.2f} N"
                    )

                else:

                    if self.ratio_mode == "weight_per_load":

                        print(f"[Iter {iteration + 1:3d}] Loss (weight/load): {loss:4.2f}")

                    else:

                        print(f"[Iter {iteration + 1:3d}] load/weight: {-loss:4.2f}")

        """
            if verbose:
                final_loss = self.loss_history[-1]
                # Debounce repeated final-summary prints within short intervals
                now = time.time()
                if now - self._last_summary_time < 0.5:
                    # skip printing; avoid spamming when train() is invoked repeatedly
                    pass
                else:
                    self._last_summary_time = now
                    if self.ratio_mode is None:
                        final_crit_load = -final_loss
                        initial_crit_load = -self.loss_history[0] if self.loss_history else 0
                        improvement_pct = ((final_crit_load - initial_crit_load) / abs(initial_crit_load) * 100) if initial_crit_load != 0 else 0
                        print(f"\nTraining complete!")
                        print(f"  Initial critical load: {initial_crit_load:.2f} N")
                        print(f"  Final critical load:   {final_crit_load:.2f} N")
                        print(f"  Improvement: {improvement_pct:.2f}%")
                    else:
                        print(f"\nTraining complete!")
                        if self.ratio_mode == 'weight_per_load':
                            print(f"  Final weight/load ratio: {final_loss:.2f}")
                            print(f"  (Lower is better = higher strength-to-weight)")
                        else:
                            print(f"  Final -load/weight (minimized): {final_loss:.2f}")
                            print(f"  (More negative is better = higher load-to-weight)")
        """

        if verbose and self.best_params:

            best_iter_display = self.best_iteration + 1

            if self.ratio_mode == "max_load_per_weight":

                best_load_weight = -self.best_objective_value

                print(
                    f"\n  Best load/weight ratio: {best_load_weight:.2f} (achieved at iteration {best_iter_display})"
                )

            elif self.ratio_mode == "weight_per_load":

                print(
                    f"\n  Best weight/load ratio: {self.best_objective_value:.2f} (achieved at iteration {best_iter_display})"
                )

            else:

                best_crit_load = -self.best_objective_value

                print(
                    f"\n  Best critical load: {best_crit_load:.2f} N (achieved at iteration {best_iter_display})"
                )

        return self.initial_params_dict, self.best_params

    def get_optimization_history(self) -> Dict:
        """Return full optimization history for visualization."""

        return {
            "loss_history": self.loss_history,
            "critical_load_history": self.critical_load_history,
            "param_history": self.param_history,
        }

    def get_param_deltas(self, initial: Dict, final: Dict) -> Dict[str, float]:
        """Compute parameter deltas: final - initial."""

        deltas = {}

        for key in initial:

            if key in final:

                if isinstance(initial[key], (int, float)) and isinstance(final[key], (int, float)):

                    deltas[key] = final[key] - initial[key]

        return deltas


def make_and_train_pratt(
    seed: int = 42,
    iterations: int = 200,
    lr: float = 0.01,
    verbose: bool = True,
    material=HighDensityBalsaWood(),
    log_interval: int = 20,
    flat_factor: float = 0.5,
    fixed_cost: float = 0.0,
) -> Tuple[Dict, Dict]:
    """Create and train a Pratt truss bridge optimizer. Returns (initial_params, best_params)."""

    def vprint(*args, **kwargs):

        if verbose:

            print(*args, **kwargs)

    vprint("=" * 70)
    vprint("Bridge Design Optimizer - Example Run")
    vprint("=" * 70)

    min = inches_to_meters(0.1)

    param_bounds = {
        "length": (inches_to_meters(18.5), inches_to_meters(18.5)),
        "angle": (
            math.atan(inches_to_meters(10.0) / (inches_to_meters(18.5) / 2)),
            math.radians(85),
        ),
        "height": (inches_to_meters(4.0), inches_to_meters(10.0)),
        "incline_thickness": (min, inches_to_meters(0.5)),
        "diagonal_thickness": (min, inches_to_meters(0.5)),
        "mid_vert_thickness": (min, inches_to_meters(0.5)),
        "side_vert_thickness": (min, inches_to_meters(0.5)),
        "top_thickness": (min, inches_to_meters(0.5)),
        "bottom_thickness": (min, inches_to_meters(0.5)),
        "incline_depth": (min, inches_to_meters(0.5)),
        "diagonal_depth": (min, inches_to_meters(0.5)),
        "mid_vert_depth": (min, inches_to_meters(0.5)),
        "side_vert_depth": (min, inches_to_meters(0.5)),
        "top_depth": (min, inches_to_meters(0.5)),
        "bottom_depth": (min, inches_to_meters(0.5)),
    }

    initial = random_initial_params(bounds=param_bounds, material=material, seed=seed)

    optimizer = BridgeOptimizer(
        initial_params=initial,
        param_bounds=param_bounds,
        learning_rate=lr,
        fixed_cost=fixed_cost,
        verbose=verbose,
    )

    init_params, final_params = optimizer.train(
        num_iterations=iterations, verbose=verbose, log_interval=log_interval
    )

    vprint("\n" + "=" * 70)
    vprint("Parameter Changes (Initial -> Final)")
    vprint("=" * 70)

    deltas = optimizer.get_param_deltas(init_params, final_params)

    for param_name in sorted(deltas.keys()):

        if deltas[param_name] != 0:

            init_val = init_params[param_name]

            final_val = final_params[param_name]

            delta = deltas[param_name]

            pct_change = (delta / init_val * 100) if init_val != 0 else 0

            vprint(
                f"  {param_name:25s}: {init_val:4.2e} → {final_val:4.2e} (Δ {delta:+.3e}, {pct_change:+.2f}%)"
            )

    return init_params, final_params
