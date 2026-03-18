import math
from abc import ABC, abstractmethod

import torch


class Member(ABC):
    @abstractmethod
    def __init__(
        self,
        length,
        thickness,
        depth,
        material,
        a: torch.Tensor,
        b: torch.Tensor,
        k,
        softmin_temperature=1e-6,
    ):
        assert all(
            type(param) == torch.Tensor for param in [length, thickness, depth, a, b]
        ), f"All parameters must be torch tensors. length thickness, depth, a, and b were recieved as types {[type(param) for param in [length, thickness, depth, a, b]]} in a member of type {type(self)}. If you are trying to create a member with fixed parameters, please provide the parameters as torch tensors."
        self.length = length
        self.thickness = thickness
        self.depth = depth
        self.material = material
        self.a = a
        self.b = b
        self.k = k
        self.softmin_temperature = softmin_temperature
        self.area = self.thickness * self.depth

        # c = distance from neutral axis to extreme fibre in plane = thickness/2
        # (all bending moments are in the plane, so the in-plane extreme fibre governs)
        self.c = self.thickness / 2

    @property
    def critical_load(self):
        return min(self.failure_modes().values())

    @abstractmethod
    def failure_modes(self, differentiable=False) -> dict:
        pass

    def volume(self, differentiable=False):
        if differentiable:
            return self.length * self.thickness * self.depth
        else:
            return self.length.item() * self.thickness.item() * self.depth.item()


class CompressionMember(Member):
    def __init__(
        self,
        length,
        thickness,
        depth,
        material,
        a: torch.Tensor,
        b: torch.Tensor,
        k,
        softmin_temperature=1e-6,
    ):
        super().__init__(
            length, thickness, depth, material, a, b, k, softmin_temperature=softmin_temperature
        )

    def failure_modes(self, differentiable=False) -> dict:
        I_in = (self.thickness**3 * self.depth) / 12
        I_out = (self.thickness * self.depth**3) / 12
        vals = torch.stack([I_in, I_out])
        weights = torch.nn.functional.softmin(vals / self.softmin_temperature, dim=0)
        I_min = (vals * weights).sum()
        if differentiable:
            return {
                "Crushing (Axial + Bending)": self.material.sigma_compression
                / ((self.a / self.area) + (torch.abs(self.b) * self.c / I_min)),
                "Buckling in Plane (Axial + Bending": (
                    self.material.E * torch.pi**2 * I_in / (self.length * self.k) ** 2
                )
                / (self.a + torch.abs(self.b) * self.c * self.area / I_in),
                "Buckling out of Plane (Axial Only)": (torch.pi**2 * self.material.E * I_out)
                / (self.a * (self.k * self.length) ** 2),
            }
        else:
            return {
                "Crushing (Axial + Bending)": self.material.sigma_compression
                / (
                    (self.a.item() / self.area.item())
                    + (abs(self.b.item()) * self.c.item() / I_min.item())
                ),
                "Buckling in Plane (Axial + Bending": (
                    self.material.E * math.pi**2 * I_in.item() / (self.length.item() * self.k) ** 2
                )
                / (
                    self.a.item()
                    + abs(self.b.item()) * self.c.item() * self.area.item() / I_in.item()
                ),
                "Buckling out of Plane (Axial Only)": (math.pi**2 * self.material.E * I_out.item())
                / (self.a.item() * (self.k * self.length.item()) ** 2),
            }


class TensionMember(Member):
    def __init__(
        self,
        length,
        thickness,
        depth,
        material,
        a: torch.Tensor,
        b: torch.Tensor,
        k,
        softmin_temperature=1e-6,
    ):
        super().__init__(
            length, thickness, depth, material, a, b, k, softmin_temperature=softmin_temperature
        )

    def failure_modes(self, differentiable=False) -> dict:
        self.I_in = (self.thickness**3 * self.depth) / 12
        if differentiable:
            return {
                "Combined Stress In Plane (Axial + Bending)": self.material.sigma_tension
                / ((self.a / self.area) + (torch.abs(self.b) * self.c / self.I_in))
            }
        else:
            return {
                "Combined Stress In Plane (Axial + Bending)": self.material.sigma_tension
                / (
                    (self.a.item() / self.area.item())
                    + (abs(self.b.item()) * self.c.item() / self.I_in.item())
                )
            }
