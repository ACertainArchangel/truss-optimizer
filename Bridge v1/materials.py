from dataclasses import dataclass


@dataclass
class Material:
    E: float
    sigma_compression: float
    sigma_tension: float
    density: float

    def __repr__(self):
        return f"{self.__class__.__name__}"


class Steel(Material):
    def __init__(self):
        super().__init__(E=200e9, sigma_compression=250e6, sigma_tension=400e6, density=7850.0)


class Aluminum(Material):
    def __init__(self):
        super().__init__(E=69e9, sigma_compression=150e6, sigma_tension=200e6, density=2700.0)


class Titanium(Material):
    def __init__(self):
        super().__init__(E=19.6e6, sigma_compression=900e6, sigma_tension=950e6, density=4500.0)


class OchromaWood(Material):
    def __init__(self):
        super().__init__(E=3.71e9, sigma_compression=11.6e6, sigma_tension=19.6e6, density=150.0)


class ConservativeBalsaWood(Material):
    def __init__(self):
        E = 2.0e9
        sigma_compression = 2.5e6
        sigma_tension = 3.0e6
        density = 240
        super().__init__(E, sigma_compression, sigma_tension, density)


class LowDensityBalsaWood(Material):
    def __init__(self):
        super().__init__(E=3e9, sigma_compression=10e6, sigma_tension=15e6, density=160.0)


class HighDensityBalsaWood(Material):
    def __init__(self):
        super().__init__(E=6e9, sigma_compression=20e6, sigma_tension=25e6, density=320.0)


class Concrete(Material):
    def __init__(self):
        super().__init__(E=30e9, sigma_compression=40e6, sigma_tension=4e6, density=2400.0)


class PolylacticAcid(Material):
    def __init__(self):
        super().__init__(E=3.5e9, sigma_compression=60e6, sigma_tension=50e6, density=1250.0)


class CarbonFiberPLA(Material):
    def __init__(self):
        super().__init__(E=10e9, sigma_compression=100e6, sigma_tension=90e6, density=1350.0)


class CustomMaterial(Material):
    def __init__(self, E: float, sigma_compression: float, sigma_tension: float, density: float):
        super().__init__(E, sigma_compression, sigma_tension, density)
