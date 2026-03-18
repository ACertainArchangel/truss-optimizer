import torch

from .trusses import BaseTruss


class Optimizer:
    def __init__(
        self, bridge, objective, learning_rate, unit_system, constraints, fixed_parameters=None
    ):
        assert objective in [
            "load_to_weight",
            "critical_load",
        ], "Objective must be either 'load_to_weight' or 'critical_load'."
        assert unit_system in [
            "metric",
            "imperial",
        ], "Unit system must be either 'metric' or 'imperial'."
        assert isinstance(constraints, dict), "Constraints must be provided as a dictionary."
        assert all(
            isinstance(constraints[key], tuple) and len(constraints[key]) == 2
            for key in constraints
        ), "Each constraint must be a tuple of (min, max)."
        for key in constraints:
            lo, hi = constraints[key]
            assert isinstance(lo, (int, float)) and isinstance(
                hi, (int, float)
            ), f"Constraint bounds must be numbers, not {type(lo)} and {type(hi)}."
        assert (
            isinstance(fixed_parameters, list) or fixed_parameters is None
        ), "Fixed parameters must be provided as a list or None."
        assert (
            bridge != "London Bridges"
        ), "FATAL ERROR: London Bridges falling down, falling down, falling down. London Bridges falling down, my fair lady."
        assert isinstance(
            bridge, BaseTruss
        ), "Bridge must be an instance of a subclass of BaseTruss."

        self.bridge = bridge
        self.objective = objective
        self.learning_rate = learning_rate
        self.unit_system = unit_system  # Materials have stuff given in pascals and density in kg/m^3 so we internally convert to metric
        self.constraints = self.bridge.expand_required_parameters(constraints)
        self.fixed_parameters = fixed_parameters or []  # Because [] as a default arg is pain

        trainable = [
            value
            for key, value in bridge.parameters.items()
            if key not in self.fixed_parameters and value.requires_grad
        ]
        if any(param.requires_grad == False for param in trainable):
            raise ValueError(
                f"Somehow non gradient parameters got into the trainable list. Names of offending parameters: {[key for key, value in bridge.parameters.items() if key not in self.fixed_parameters and value.requires_grad == False]}"
            )
        self.adam = torch.optim.Adam(trainable, lr=learning_rate)

    def step(self):
        self.bridge.members = self.bridge.generate_members()
        self.adam.zero_grad()

        if self.objective == "load_to_weight":
            loss = -self.bridge.load_to_weight_ratio(differentiable=True)
        elif self.objective == "critical_load":
            loss = -self.bridge.critical_load(differentiable=True)
        else:
            raise ValueError(
                f"Unsupported objective: {self.objective}. This was not suppoed to happen since we had an error check for this before so either you are trying weird monkey patching or my code broke somewhere."
            )

        loss.backward()
        self.adam.step()

        with torch.no_grad():
            self.bridge.clamp_params_to_valid_ranges(
                min_top_size_in_meters=0.1524
            )  # TODO: THIS IS AN EXCEPTIONALLY BAD HACK AND THIS NEEDS TO BE MADE BETTER!!! THIS ENTIRE BLOCK NEEDS TO BE ONE METHOD AND WORK FROM CONSTRAINTS.
            for name, param in self.bridge.parameters.items():
                if name in self.constraints:
                    lo, hi = self.constraints[name]
                    param.clamp_(lo + 1e-3, hi - 1e-3)
                    if name in self.fixed_parameters:
                        assert (
                            lo <= param.item() <= hi
                        ), f"Fixed parameter '{name}' with value {param.item()} is out of bounds ({lo}, {hi}). Please provide a value within the specified constraints for this parameter."
            self.bridge.clamp_params_to_valid_ranges(
                min_top_size_in_meters=0.1524
            )  # TODO: THIS IS AN EXCEPTIONALLY BAD HACK AND THIS NEEDS TO BE MADE BETTER!!! THIS ENTIRE BLOCK NEEDS TO BE ONE METHOD AND WORK FROM CONSTRAINTS.

        return (
            self.bridge.critical_load(differentiable=False)
            if self.objective == "critical_load"
            else self.bridge.load_to_weight_ratio(differentiable=False)
        )

    def optimize(self, iterations, verbose=False):
        objectives = []
        for i in range(iterations):
            objectives.append(self.step())
            (
                print(
                    f"Iteration {i+1}/{iterations} - {'Load to Weight Ratio' if self.objective == 'load_to_weight' else 'Critical Load'}: {objectives[-1]:.4f}"
                )
                if verbose and (i + 1) % 100 == 0
                else None
            )

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(objectives)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(
            "Load to Weight Ratio" if self.objective == "load_to_weight" else "Critical Load"
        )
        ax.set_title(
            f"Optimization of {self.objective.replace('_', ' ').title().capitalize()} over {iterations} Iterations"
        )
        ax.grid()

        # Add a line
        best = max(objectives)
        best_iter = objectives.index(best)
        ax.axvline(best_iter, color="red", linestyle="--", label=f"Best at iter {best_iter}")
        ax.legend()

        plt.close(fig)
        return fig
