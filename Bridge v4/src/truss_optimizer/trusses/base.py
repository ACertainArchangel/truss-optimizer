import random as rand
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TypedDict

import torch
from torch.nn.functional import softmin

from .members import CompressionMember, TensionMember


class BaseTruss(ABC):

    # This thing gets called as super().init() and then we generate the members list in the child class init.
    def __init__(
        self,
        material,
        length,
        constraints=None,
        parameters=None,
        k=1.0,
        softmin_temperature=1e-6,
        fixed_cost=None,
    ):
        if not (constraints is None) ^ (parameters is None):
            raise ValueError("Exactly one of 'constraints' or 'parameters' must be provided.")
        self.material = material
        self.length = length
        self.softmin_temperature = softmin_temperature
        self.k = k
        self.fixed_cost = torch.tensor(fixed_cost) if fixed_cost is not None else torch.tensor(0.0)
        if constraints:
            assert (
                parameters is None
            ), "Parameters should not be provided when constraints are given."
            parameters = {}
            for key in constraints:
                if key in [
                    "compression_member_thickness",
                    "compression_member_depth",
                    "tension_member_thickness",
                    "tension_member_depth",
                ]:
                    for param in self.required_parameters[key]:
                        parameters[param] = torch.tensor(
                            rand.uniform(constraints[key][0], constraints[key][1]),
                            requires_grad=True,
                        )
                elif key == "misc":
                    ValueError(
                        "The 'misc' category in required_parameters is just for organizational purposes. Please provide the parameters in 'misc' directly in the parameters dictionary with their own keys, not grouped under a 'misc' key. This error is to keep you from trying to define angle and height as if they were in the same units. If you believe you really need to define height and angle in the same line, please call this number: 1-800-ICE-COLD and ask to speak to Gabe about it. He will be very impressed by your dedication to bad ideas and will be happy to help you figure out how to do it."
                    )
                else:
                    parameters[key] = torch.tensor(
                        rand.uniform(constraints[key][0], constraints[key][1]), requires_grad=True
                    )
            self.parameters = parameters
        elif parameters:
            for key in parameters:
                if key in [
                    "compression_member_thickness",
                    "compression_member_depth",
                    "tension_member_thickness",
                    "tension_member_depth",
                ]:
                    for param in self.required_parameters[key]:
                        parameters[param] = torch.tensor(parameters[param], requires_grad=True)
                        import warnings

                        warnings.warn(
                            f"Initialising all {key} parameters with value {parameters[param]}? Ice cold move. (Bad idea)."
                        )
                elif key == "misc":
                    ValueError(
                        "The 'misc' category in required_parameters is just for organizational purposes. Please provide the parameters in 'misc' directly in the parameters dictionary with their own keys, not grouped under a 'misc' key. This error is to keep you from trying to define angle and height as if they were in the same units. If you believe you really need to define height and angle in the same line, please call this number: 1-800-ICE-COLD and ask to speak to Gabe about it. He will be very impressed by your dedication to bad ideas and will be happy to help you figure out how to do it."
                    )
                else:
                    if not isinstance(parameters[key], torch.Tensor):
                        parameters[key] = torch.tensor(parameters[key], requires_grad=True)
            self.parameters = parameters
            assert (
                constraints is None
            ), "Constraints should not be provided when parameters are given."

        flattened_required_params = [
            param for sublist in self.required_parameters.values() for param in sublist
        ]
        assert all(
            param in self.parameters for param in flattened_required_params
        ), f"Missing required parameters. Required parameters are {flattened_required_params}, but got {list(self.parameters.keys())}.\n\n This means we are missing {[param for param in flattened_required_params if param not in self.parameters]} \n\n and extra parameters: {[param for param in self.parameters if param not in flattened_required_params]}."

        assert all(
            param.requires_grad for key, param in self.parameters.items()
        ), "All parameters must require gradients for optimization. If you want to use fixed parameters, please provide them as torch tensors with requires_grad=True but exclude them from the optimizer's trainable parameters by listing them in the fixed_parameters argument of the Optimizer class. Also this was supposed to get patched internally so if you are seeing this error message go find Gabe and tell him he is an idiot."

        self.clamp_params_to_valid_ranges()

        self.members = self.generate_members()

        try:
            for member in self.members:
                if member.length.item() <= 0:
                    raise ValueError(
                        f"Initial design has a member with non-positive length ({member.length.item():.4f} m). The geometry is impossible — check that height and angle constraints are compatible with the bridge span."
                    )

                if self.critical_load() <= 0:
                    raise ValueError(
                        f"Initial design is not viable with a critical load of {self.critical_load():.2f} N. Please adjust your initial parameters or constraints to create a viable design before optimization."
                    )
        except Exception as e:
            if hasattr(self, "constraints"):
                from warnings import warn

                warn("Bro you designed an invalid bridge.")
                raise e
            else:
                from warnings import warn

                warn("Bro your constraints allow illegal bridges.")
                raise e

    ##########################################################################

    class RequiredParameters(TypedDict):
        compression_member_thickness: list[str]
        compression_member_depth: list[str]
        tension_member_thickness: list[str]
        tension_member_depth: list[str]
        misc: list[str]

    ##########################################################################

    # Names of all the parameters that the bridge is gonna need to generate its members
    @property
    @abstractmethod
    def required_parameters(self) -> RequiredParameters:
        pass

    # Generates a list of members that use the tensor parameters
    @abstractmethod
    def generate_members(self) -> list:
        pass

    # Returns an image of the bridge and keeps the animation without auto closing if keep_animation_up is True. I could plug in a CAD renderer later.
    @abstractmethod
    def visualize(self, keep_animation_up=False):
        pass

    # Write a big string about the bridge.
    @abstractmethod
    def generate_report(self):
        pass

    @abstractmethod
    def clamp_params_to_valid_ranges(self):
        pass

    ##########################################################################

    def expand_required_parameters(self, constraints):
        expanded_params = {}
        rp = self.required_parameters
        for key, value in constraints.items():
            if key in rp and isinstance(rp[key], list):
                for param in rp[key]:
                    expanded_params[param] = value
            else:
                expanded_params[key] = value
        return expanded_params

    def rounded(self, numerator, denominator, rounding=None):
        assert (
            rounding is not None
        ), "Rounding must be specified as a list of parameter names to round, or 'all' to round all parameters. Supported parameters are 'tension_member_depth', 'compression_member_depth', 'tension_member_thickness', and 'compression_member_thickness'."
        if rounding == "all":
            rounding = [
                "tension_member_depth",
                "compression_member_depth",
                "tension_member_thickness",
                "compression_member_thickness",
            ]

        newbridge = self.__class__(
            material=self.material,
            length=self.length,
            parameters={param: self.parameters[param].item() for param in self.parameters},
            k=self.k,
            softmin_temperature=self.softmin_temperature,
        )

        for member in newbridge.members:
            if isinstance(member, CompressionMember):
                if "compression_member_depth" in rounding:
                    member.depth = torch.tensor(
                        round(member.depth.item() * denominator / numerator)
                        * numerator
                        / denominator,
                        requires_grad=True,
                    )
                if "compression_member_thickness" in rounding:
                    member.thickness = torch.tensor(
                        round(member.thickness.item() * denominator / numerator)
                        * numerator
                        / denominator,
                        requires_grad=True,
                    )
            elif isinstance(member, TensionMember):
                if "tension_member_depth" in rounding:
                    member.depth = torch.tensor(
                        round(member.depth.item() * denominator / numerator)
                        * numerator
                        / denominator,
                        requires_grad=True,
                    )
                if "tension_member_thickness" in rounding:
                    member.thickness = torch.tensor(
                        round(member.thickness.item() * denominator / numerator)
                        * numerator
                        / denominator,
                        requires_grad=True,
                    )
            else:
                raise ValueError(
                    f"Unsupported member type: {type(member)}. How on eath did you get a {type(member)} in your members list? A {type(member)}? Seriously? {type(member)}'s should not be in your members list. They belong elsewhere. Like in lists of {type(member)}'s. Not in lists of members. Honestly. Programmers these days and their {type(member)}'s."
                )  # This is a joke and should never happen.

        return newbridge

    def load_to_weight_ratio(self, differentiable=False):
        if differentiable:
            return self.critical_load(differentiable=True) / self.weight(differentiable=True)
        else:
            return self.critical_load(differentiable=False) / self.weight(differentiable=False)

    def critical_load(self, differentiable=False):
        if differentiable:
            vals = torch.stack(
                [
                    tensor
                    for member in self.members
                    for tensor in member.failure_modes(differentiable=True).values()
                ]
            )
            weights = softmin(vals / self.softmin_temperature, dim=0)
            return (weights * vals).sum()
        else:
            return min(
                min(member.failure_modes(differentiable=False).values()) for member in self.members
            )

    def volume(self, differentiable=False):
        return sum(member.volume(differentiable=differentiable) for member in self.members)

    def weight(self, differentiable=False):
        return (
            self.volume(differentiable=differentiable) * self.material.density * 9.81
            + self.fixed_cost
        )
