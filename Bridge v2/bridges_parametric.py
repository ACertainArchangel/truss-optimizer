import math
from math import cos, sin, tan


class Bridge:

    def __init__(self, material=None):
        self.material = material

    class TensionMember:

        def __init__(self, ultimate_tensile_str, cross_sec_area, M_max=0, depth=None, length=None):
            """Initialize tension member with optional bending moment parameters."""

            self.ultimate_tensile_str = ultimate_tensile_str
            self.cross_sec_area = cross_sec_area
            self.area = self.cross_sec_area
            self.M_max = M_max
            self.depth = depth
            self.length = length

        @property
        def volume(self):
            """Volume = cross_sec_area * length"""

            if self.length is not None:

                return self.cross_sec_area * self.length

            return 0.0

        def get_failure_force(self):
            """Maximum axial force before tension failure."""

            return self.cross_sec_area * self.ultimate_tensile_str

        def combined_stress_failure_ratio(self, F_axial):
            """Stress ratio (< 1 means safe). Includes optional bending."""

            sigma = F_axial / self.cross_sec_area

            if self.M_max and self.depth:
                thickness = self.cross_sec_area / self.depth
                I = self.depth * thickness**3 / 12
                c = thickness / 2
                sigma += self.M_max * c / I

            return sigma / self.ultimate_tensile_str

        def find_failure_F(self, axial_lambda, moment_lambda=None):
            """Find applied load F causing combined stress failure (ratio = 1)."""

            from scipy.optimize import fsolve

            def failure_equation(F):

                F_axial = axial_lambda(F)
                M_max = moment_lambda(F) if moment_lambda else 0
                sigma = F_axial / self.cross_sec_area

                if M_max and self.depth:
                    thickness = self.cross_sec_area / self.depth
                    I = self.depth * thickness**3 / 12
                    c = thickness / 2
                    sigma += abs(M_max) * c / I

                return sigma / self.ultimate_tensile_str - 1.0

            initial_guesses = [
                self.get_failure_force(),
                self.get_failure_force() / 2,
                self.get_failure_force() * 2,
                self.get_failure_force() / 10,
            ]

            pos_roots = []

            for guess in initial_guesses:

                try:

                    root = fsolve(failure_equation, guess)[0]

                    if root > 1e-6:

                        pos_roots.append(root)
                except:
                    continue

            if pos_roots:
                return min(pos_roots)

            else:
                return abs(fsolve(failure_equation, self.get_failure_force())[0])

        def get_volume(self, len):
            return len * self.cross_sec_area

    class CompressionMember:

        def __init__(self, length, thickness, depth, K, E, sigma_ult):
            """Initialize compression member with geometry, buckling factor K, and material props."""

            self.length = length
            self.thickness = thickness
            self.depth = depth
            self.K = K
            self.E = E
            self.sigma_ult = sigma_ult
            self.cross_sec_area = thickness * depth
            self.A = thickness * depth
            self.area = self.A
            self.I_in_plane = depth * thickness**3 / 12
            self.I_out_of_plane = thickness * depth**3 / 12
            self.c = thickness / 2
            self.volume = self.length * self.thickness * self.depth

        def max_compressive_force_in_plane(self, is_in_plane=True):
            """Euler buckling limit."""

            F_cr = (
                (3.14159265**2)
                * self.E
                * (self.I_in_plane if is_in_plane else self.I_out_of_plane)
                / (self.K * self.length) ** 2
            )

            return F_cr

        def combined_stress_failure_ratio(self, F_axial, M_max=0, is_in_plane=True):
            """Stress ratio (< 1 means safe). Combined axial + bending."""

            sigma = F_axial / self.A + M_max * self.c / (
                self.I_in_plane if is_in_plane else self.I_out_of_plane
            )

            return sigma / self.sigma_ult

        def find_euler_buckle_F(self, axial_lambda, moment_lambda=None, is_in_plane=True):
            """Find applied load F causing Euler buckling with optional moment interaction."""

            from scipy.optimize import fsolve

            F_cr_member = self.max_compressive_force_in_plane(is_in_plane)

            if moment_lambda is None:
                F_euler = F_cr_member / abs(axial_lambda(1.0))

                return F_euler

            def buckling_interaction_equation(F):
                F_axial = axial_lambda(F)
                M_max = moment_lambda(F)
                I = self.I_in_plane if is_in_plane else self.I_out_of_plane
                c = self.c
                ratio = F_axial / F_cr_member + (M_max * c * self.A) / (F_cr_member * I)

                return ratio - 1.0

            initial_guess = F_cr_member / (2.0 * abs(axial_lambda(1.0)))

            try:
                F_euler_with_moment = fsolve(buckling_interaction_equation, initial_guess)[0]
            except:
                F_euler_with_moment = F_cr_member / abs(axial_lambda(1.0))

            return F_euler_with_moment

        def find_material_strength_F(self, axial_lambda, moment_lambda=None, is_in_plane=True):
            """Find applied load F causing material strength failure (combined stress)."""

            from scipy.optimize import fsolve

            def material_failure_equation(F):
                F_axial = axial_lambda(F)
                M_max = moment_lambda(F) if moment_lambda else 0
                sigma = F_axial / self.A

                if M_max:

                    I = self.I_in_plane if is_in_plane else self.I_out_of_plane

                    c = self.c

                    sigma += M_max * c / I

                return sigma / self.sigma_ult - 1.0

            initial_guess = self.A * self.sigma_ult / abs(axial_lambda(1.0))

            try:
                F_material = fsolve(material_failure_equation, initial_guess)[0]

            except:
                F_material = float("inf")

            return F_material

        def find_buckle_F(self, axial_lambda, moment_lambda=None, is_in_plane=True):
            """Find applied load F causing buckling: min of Euler buckling and material strength."""

            F_euler = self.find_euler_buckle_F(axial_lambda, moment_lambda, is_in_plane)
            F_material = self.find_material_strength_F(axial_lambda, moment_lambda, is_in_plane)

            return min(F_euler, F_material)


class OnePanelPratt2D(Bridge):
    def __init__(
        self,
        angle,
        height,
        length,
        incline_thickness,
        diagonal_thickness,
        mid_vert_thickness,
        side_vert_thickness,
        top_thickness,
        bottom_thickness,
        incline_depth,
        diagonal_depth,
        mid_vert_depth,
        side_vert_depth,
        top_depth,
        bottom_depth,
        E,
        sigma_compression,
        sigma_tension,
        material=None,
        K=1.0,
    ):

        super().__init__(material=material)

        self.angle = angle
        self.height = height
        self.length = length

        self.phi = math.atan(((self.length - 2 * height / tan(angle)) / 2) / (height))

        self.members = {
            "incline": self.CompressionMember(
                length=height / sin(angle),
                thickness=incline_thickness,
                depth=incline_depth,
                K=K,
                E=E,
                sigma_ult=sigma_compression,
            ),
            "diagonal": self.TensionMember(
                ultimate_tensile_str=sigma_tension,
                cross_sec_area=diagonal_thickness * diagonal_depth,
                depth=diagonal_depth,
                length=height / cos(self.phi),
            ),
            "top_chord": self.CompressionMember(
                length=length - (2 * height / tan(angle)),
                thickness=top_thickness,
                depth=top_depth,
                K=K,
                E=E,
                sigma_ult=sigma_compression + 100000000000,
            ),  # Turning off top chord faulure governance with a temporary HACK.
            "bottom_chord": self.TensionMember(
                ultimate_tensile_str=sigma_tension,
                cross_sec_area=bottom_thickness * bottom_depth,
                depth=bottom_depth,
                length=length,
            ),
            "mid_vert": self.CompressionMember(
                length=height,
                thickness=mid_vert_thickness,
                depth=mid_vert_depth,
                K=K,
                E=E,
                sigma_ult=sigma_compression,
            ),
            "side_vert": self.CompressionMember(
                length=height,
                thickness=side_vert_thickness,
                depth=side_vert_depth,
                K=K,
                E=E,
                sigma_ult=sigma_compression,
            ),
        }

    def get_axial_forces_from_F_dict(self):

        axial_forces = {}
        axial_forces["incline"] = lambda F: F / (2 * sin(self.angle))
        axial_forces["diagonal"] = lambda F: F / (6 * cos(self.phi))
        axial_forces["top_chord"] = lambda F: axial_forces["incline"](F) * cos(
            self.angle
        ) + axial_forces["diagonal"](F) * sin(self.phi)
        axial_forces["bottom_chord"] = axial_forces["top_chord"]
        total_vertical_membs_area = (
            2 * self.members["side_vert"].area + self.members["mid_vert"].area
        )
        total_force_covered_by_vert_membs = (
            lambda F: F
            - axial_forces["incline"](F) * sin(self.angle) * 2
            + axial_forces["diagonal"](F) * cos(self.phi) * 2
        )
        axial_forces["mid_vert"] = (
            lambda F_axial: self.members["mid_vert"].area
            / total_vertical_membs_area
            * total_force_covered_by_vert_membs(F_axial)
        )
        axial_forces["side_vert"] = (
            lambda F_axial: self.members["side_vert"].area
            / total_vertical_membs_area
            * total_force_covered_by_vert_membs(F_axial)
        )
        return axial_forces

    def get_required_F_from_axial_forces_dict(self):
        """Gives the inverse of get_axial_forces_formulas"""

        load_formulas = {}

        load_formulas["incline"] = lambda F_axial: 2 * F_axial * sin(self.angle)

        load_formulas["diagonal"] = lambda F_axial: 6 * F_axial * cos(self.phi)

        load_formulas["top_chord"] = lambda F_axial: F_axial / (
            cos(self.angle) / (2 * sin(self.angle)) + sin(self.phi) / (6 * cos(self.phi))
        )

        load_formulas["bottom_chord"] = load_formulas["top_chord"]

        total_vertical_membs_area = (
            2 * self.members["side_vert"].area + self.members["mid_vert"].area
        )

        load_formulas["mid_vert"] = (
            lambda F_axial: 3 * F_axial * total_vertical_membs_area / self.members["mid_vert"].area
        )

        load_formulas["side_vert"] = (
            lambda F_axial: 3 * F_axial * total_vertical_membs_area / self.members["side_vert"].area
        )

        return load_formulas

    def get_moments_from_F_dict(self):
        moments = {}
        moments["incline"] = [lambda F: 0.0]
        moments["diagonal"] = [lambda F: 0.0]

        axial_dict = self.get_axial_forces_from_F_dict()

        side_of_top_vertical_force = (
            lambda F: axial_dict["incline"](F) * sin(self.angle)
            + axial_dict["side_vert"](F)
            - axial_dict["diagonal"](F) * cos(self.phi)
            - F / 3
        )

        moment_at_center_top = lambda F: side_of_top_vertical_force(F) * (
            self.length / 2 - self.height / tan(self.angle)
        )

        moment_at_end_top = lambda F: side_of_top_vertical_force(F) * (
            self.length - 2 * self.height / tan(self.angle)
        ) + (axial_dict["mid_vert"](F) - F / 3) * (self.length / 2 - self.height / tan(self.angle))

        assert math.isclose(
            side_of_top_vertical_force(1.0) * 2 + axial_dict["mid_vert"](1.0) - 1.0 / 3,
            0.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ), f"Debug: vertical forces on top chord do not sum to zero! In fact they sum to {side_of_top_vertical_force(1.0)*2+axial_dict['mid_vert'](1.0)-1.0/3}"

        moments["top_chord"] = [moment_at_center_top, moment_at_end_top]
        side_of_bottom_vertical_force = lambda F: F / 2 - axial_dict["incline"](F) * sin(self.angle)
        intermediate_bottom_force = lambda F: -axial_dict["side_vert"](F)
        center_bottom_force = lambda F: 2 * axial_dict["diagonal"](F) * cos(self.phi) - axial_dict[
            "mid_vert"
        ](F)

        moment1_bottom = lambda F: side_of_bottom_vertical_force(F) * self.height / tan(self.angle)

        moment2_bottom = lambda F: side_of_bottom_vertical_force(
            F
        ) * self.length / 2 + intermediate_bottom_force(F) * (
            self.length / 2 - self.height / tan(self.angle)
        )

        moment3_bottom = (
            lambda F: side_of_bottom_vertical_force(F)
            * (self.length - self.height / tan(self.angle))
            + intermediate_bottom_force(F) * (self.length - 2 * self.height / tan(self.angle))
            + center_bottom_force(F) * (self.length / 2 - self.height / tan(self.angle))
        )

        moment4_bottom = (
            lambda F: side_of_bottom_vertical_force(F) * self.length
            + intermediate_bottom_force(F) * (self.length - self.height / tan(self.angle))
            + center_bottom_force(F) * (self.length / 2)
            + intermediate_bottom_force(F) * (self.height / tan(self.angle))
        )

        assert math.isclose(
            side_of_bottom_vertical_force(1.0) * 2
            + intermediate_bottom_force(1.0) * 2
            + center_bottom_force(1.0),
            0.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ), f"Debug: vertical forces on bottom chord do not sum to zero! In fact they sum to {side_of_bottom_vertical_force(1.0)*2+intermediate_bottom_force(1.0)*2+center_bottom_force(1.0)}. The entire expression is {side_of_bottom_vertical_force(1.0)}*2 + {intermediate_bottom_force(1.0)}*2 + {center_bottom_force(1.0)}"

        moments["bottom_chord"] = [moment1_bottom, moment2_bottom, moment3_bottom, moment4_bottom]
        moments["mid_vert"] = [lambda F: 0.0]
        moments["side_vert"] = [lambda F: 0.0]

        return moments

    def get_required_F_from_moments_dict(self):
        from scipy.optimize import fsolve

        load_from_moments = {}

        moment_fns_dict = self.get_moments_from_F_dict()

        def is_structurally_zero(moment_fn):
            """Test if a moment function is effectively zero across a range of F values"""
            test_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
            for F in test_values:
                if abs(moment_fn(F)) > 1e-9:
                    return False
            return True

        def initial_guess(M):
            return 1.0 if abs(M) < 1e-6 else M / 100.0

        def create_inverse_fn(moment_fn):
            """Create an inverse function for a moment, or return None if structurally zero"""

            if is_structurally_zero(moment_fn):
                return None
            else:
                return lambda M: fsolve(lambda F: moment_fn(F) - M, initial_guess(M))[0]
            
        top_moment_fns = moment_fns_dict["top_chord"]
        load_from_moments["top_chord"] = create_inverse_fn(top_moment_fns[0])
        bottom_moment_fns = moment_fns_dict["bottom_chord"]
        load_from_moments["bottom_chord"] = create_inverse_fn(bottom_moment_fns[1])
        load_from_moments["incline"] = None
        load_from_moments["diagonal"] = None
        load_from_moments["mid_vert"] = None
        load_from_moments["side_vert"] = None
        return load_from_moments

    def get_failure_mode_dict(self):
        req_F_from_axial = self.get_required_F_from_axial_forces_dict()
        req_F_from_moments = self.get_required_F_from_moments_dict()

        f_mode_dict = {}

        for name, member in self.members.items():

            axial_forward = self.get_axial_forces_from_F_dict()
            moments_forward = self.get_moments_from_F_dict()

            f_mode_dict = {}

            for name, member in self.members.items():
                axial_lambda = axial_forward.get(name)
                moment_lambda = None
                mf = moments_forward.get(name)

                if mf:
                    if isinstance(mf, list):
                        for fn in mf:
                            try:
                                if abs(fn(1.0)) > 1e-12:
                                    moment_lambda = fn
                                    break
                            except Exception:
                                continue
                    elif callable(mf): # THis is a hack to be able to use constants and callables.
                        moment_lambda = mf

                if isinstance(member, self.TensionMember):
                    f_mode_dict[f"{name}_rupture"] = member.find_failure_F(
                        axial_lambda=axial_lambda,
                        moment_lambda=moment_lambda,
                    )
                elif isinstance(member, self.CompressionMember):
                    f_mode_dict[f"{name}_buckle"] = member.find_euler_buckle_F(
                        axial_lambda=axial_lambda,
                        moment_lambda=moment_lambda,
                        is_in_plane=True,
                    )

                    f_mode_dict[f"{name}_buckle_out_of_plane"] = member.find_euler_buckle_F(
                        axial_lambda=axial_lambda,
                        moment_lambda=moment_lambda,
                        is_in_plane=False,
                    )

                    f_mode_dict[f"{name}_combined_stress"] = member.find_material_strength_F(
                        axial_lambda=axial_lambda,
                        moment_lambda=moment_lambda,
                        is_in_plane=True,
                    )
                else:
                    raise ValueError(f"Unknown member type for {name}")
            f_mode_dict["torsion_failure"] = None
            return f_mode_dict

    def get_total_volume(self):
        total_volume = 0.0
        total_volume += 2 * self.members["incline"].volume
        total_volume += 2 * self.members["diagonal"].volume
        total_volume += self.members["top_chord"].volume
        total_volume += self.members["bottom_chord"].volume
        total_volume += self.members["mid_vert"].volume
        total_volume += 2 * self.members["side_vert"].volume
        return total_volume
