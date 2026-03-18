import torch

DTYPE = torch.float64


def softmin_torch(values, softness=1e-3):
    """Differentiable min approximation via log-sum-exp. Filters NaN/inf."""

    v = torch.stack(values) if isinstance(values, (list, tuple)) else values
    v_finite = v[torch.isfinite(v)]

    if v_finite.numel() == 0:
        return torch.tensor(float("inf"), dtype=v.dtype, device=v.device)

    return -softness * torch.logsumexp(-v_finite / softness, dim=0)


def max_load_torch(
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
    K=1.0,
    softness=1e-6,
    dtype=DTYPE,
    device=None,
):
    """Compute differentiable max applied load for a OnePanelPratt2D truss using PyTorch."""

    def T(x):

        if isinstance(x, torch.Tensor):

            return x.to(dtype=dtype, device=device)

        return torch.tensor(x, dtype=dtype, device=device)

    angle = T(angle)

    height = T(height)

    length = T(length)

    E = T(E)

    sigma_compression = T(sigma_compression)

    sigma_tension = T(sigma_tension)

    K = T(K)

    incline_thickness = T(incline_thickness)

    diagonal_thickness = T(diagonal_thickness)

    mid_vert_thickness = T(mid_vert_thickness)

    side_vert_thickness = T(side_vert_thickness)

    top_thickness = T(top_thickness)

    bottom_thickness = T(bottom_thickness)

    incline_depth = T(incline_depth)

    diagonal_depth = T(diagonal_depth)

    mid_vert_depth = T(mid_vert_depth)

    side_vert_depth = T(side_vert_depth)

    top_depth = T(top_depth)

    bottom_depth = T(bottom_depth)

    A_incline = incline_thickness * incline_depth

    A_diagonal = diagonal_thickness * diagonal_depth

    A_mid = mid_vert_thickness * mid_vert_depth

    A_side = side_vert_thickness * side_vert_depth

    A_top = top_thickness * top_depth

    A_bottom = bottom_thickness * bottom_depth

    I_incline_in = incline_depth * incline_thickness**3 / 12.0

    I_diagonal_in = diagonal_depth * diagonal_thickness**3 / 12.0

    I_mid_in = mid_vert_depth * mid_vert_thickness**3 / 12.0

    I_side_in = side_vert_depth * side_vert_thickness**3 / 12.0

    I_top_in = top_depth * top_thickness**3 / 12.0

    I_bottom_in = bottom_depth * bottom_thickness**3 / 12.0

    I_incline_out = incline_thickness * incline_depth**3 / 12.0

    I_diagonal_out = diagonal_thickness * diagonal_depth**3 / 12.0

    I_mid_out = mid_vert_thickness * mid_vert_depth**3 / 12.0

    I_side_out = side_vert_thickness * side_vert_depth**3 / 12.0

    I_top_out = top_thickness * top_depth**3 / 12.0

    I_bottom_out = bottom_thickness * bottom_depth**3 / 12.0

    phi = torch.atan(((length - 2 * height / torch.tan(angle)) / 2) / (height))

    L_incline = height / torch.sin(angle)

    L_diagonal = height / torch.cos(phi)

    L_top = length - 2 * height / torch.tan(angle)

    L_bottom = length

    L_mid_vert = height

    L_side_vert = height

    a_incline = 1.0 / (2.0 * torch.sin(angle))

    a_diagonal = 1.0 / (6.0 * torch.cos(phi))

    a_top = a_incline * torch.cos(angle) + a_diagonal * torch.sin(phi)

    a_bottom = a_top

    total_vertical_area = 2.0 * A_side + A_mid

    a_mid = (A_mid / total_vertical_area) * (1.0 / 3.0)

    a_side = (A_side / total_vertical_area) * (1.0 / 3.0)

    lever_top = length / 2.0 - height / torch.tan(angle)

    b_top_center = a_side * lever_top

    b_bottom_2 = -a_side * lever_top

    failure_loads = {}

    def tension_Fcrit_axial_only(a_coeff, area, sigma):
        """Tension rupture from axial stress only: sigma = F*a/A"""

        a_safe = torch.clamp(torch.abs(a_coeff), min=1e-12)

        return (area * sigma) / a_safe

    def tension_Fcrit_combined(a_coeff, area, b_coeff, c, I, sigma):
        """Tension rupture from combined stress: sigma = F*a/A + F*b*c/I"""

        denom = torch.clamp(torch.abs(a_coeff) / area + torch.abs(b_coeff) * c / I, min=1e-12)

        return sigma / denom

    def euler_buckling_Fcrit(E, I, K, L, a_coeff, area, b_coeff=0, c=0):
        """Euler buckling F_crit with optional moment interaction."""

        L_safe = torch.clamp(L, min=1e-12)

        F_cr_member = (torch.pi**2 * E * I) / (K * L_safe) ** 2

        a_safe = torch.clamp(torch.abs(a_coeff), min=1e-12)

        if isinstance(b_coeff, (int, float)) and b_coeff == 0:

            return F_cr_member / a_safe

        b_abs = (
            torch.abs(b_coeff)
            if isinstance(b_coeff, torch.Tensor)
            else torch.abs(torch.tensor(b_coeff, dtype=a_coeff.dtype, device=a_coeff.device))
        )

        denom = a_safe + b_abs * c * area / I

        denom_safe = torch.clamp(denom, min=1e-12)

        return F_cr_member / denom_safe

    def compression_combined_stress_Fcrit(a_coeff, area, b_coeff, c, I, sigma):
        """Compression failure from combined stress: sigma = F*a/A + F*b*c/I"""

        denom = torch.clamp(torch.abs(a_coeff) / area + torch.abs(b_coeff) * c / I, min=1e-12)

        return sigma / denom

    def compression_combined_stress_Fcrit_no_moment(a_coeff, area, sigma):
        """Compression failure from axial stress only: sigma = F*a/A"""

        denom = torch.clamp(torch.abs(a_coeff) / area, min=1e-12)

        return sigma / denom

    failure_loads["diagonal_rupture"] = tension_Fcrit_axial_only(
        a_diagonal, A_diagonal, sigma_tension
    )

    c_bottom = bottom_thickness / 2.0

    failure_loads["bottom_chord_rupture"] = tension_Fcrit_combined(
        a_bottom, A_bottom, b_bottom_2, c_bottom, I_bottom_in, sigma_tension
    )

    failure_loads["incline_buckle"] = euler_buckling_Fcrit(
        E, I_incline_in, K, L_incline, a_incline, A_incline
    )

    failure_loads["incline_buckle_out_of_plane"] = euler_buckling_Fcrit(
        E, I_incline_out, K, L_incline, a_incline, A_incline
    )

    failure_loads["incline_combined_stress"] = compression_combined_stress_Fcrit_no_moment(
        a_incline, A_incline, sigma_compression
    )

    c_top = top_thickness / 2.0

    failure_loads["top_chord_buckle"] = euler_buckling_Fcrit(
        E, I_top_in, K, L_top, a_top, A_top, b_top_center, c_top
    )

    failure_loads["top_chord_buckle_out_of_plane"] = euler_buckling_Fcrit(
        E, I_top_out, K, L_top, a_top, A_top, b_top_center, c_top
    )

    failure_loads["top_chord_combined_stress"] = compression_combined_stress_Fcrit(
        a_top, A_top, b_top_center, c_top, I_top_in, sigma_compression
    )

    failure_loads["mid_vert_buckle"] = euler_buckling_Fcrit(
        E, I_mid_in, K, L_mid_vert, a_mid, A_mid
    )

    failure_loads["mid_vert_buckle_out_of_plane"] = euler_buckling_Fcrit(
        E, I_mid_out, K, L_mid_vert, a_mid, A_mid
    )

    failure_loads["mid_vert_combined_stress"] = compression_combined_stress_Fcrit_no_moment(
        a_mid, A_mid, sigma_compression
    )

    failure_loads["side_vert_buckle"] = euler_buckling_Fcrit(
        E, I_side_in, K, L_side_vert, a_side, A_side
    )

    failure_loads["side_vert_buckle_out_of_plane"] = euler_buckling_Fcrit(
        E, I_side_out, K, L_side_vert, a_side, A_side
    )

    failure_loads["side_vert_combined_stress"] = compression_combined_stress_Fcrit_no_moment(
        a_side, A_side, sigma_compression
    )

    vals = list(failure_loads.values())

    return softmin_torch(vals, softness=softness), failure_loads


if __name__ == "__main__":

    import math

    params = dict(
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

    top_thickness_t = torch.tensor(0.02, dtype=DTYPE, requires_grad=True)

    max_load, floads = max_load_torch(**{**params, "top_thickness": top_thickness_t})

    print("max_load", max_load.item())

    max_load.backward()

    print("d max_load / d top_thickness =", top_thickness_t.grad.item())


def compute_volume_torch(
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
    length,
    height,
    angle,
    dtype=DTYPE,
    device=None,
):
    """Compute total volume of OnePanelPratt2D truss analytically (differentiable)."""

    def T(x):

        if isinstance(x, torch.Tensor):

            return x.to(dtype=dtype, device=device)

        return torch.tensor(x, dtype=dtype, device=device)

    thickness_dict = {
        "incline": T(incline_thickness),
        "diagonal": T(diagonal_thickness),
        "mid_vert": T(mid_vert_thickness),
        "side_vert": T(side_vert_thickness),
        "top_chord": T(top_thickness),
        "bottom_chord": T(bottom_thickness),
    }

    depth_dict = {
        "incline": T(incline_depth),
        "diagonal": T(diagonal_depth),
        "mid_vert": T(mid_vert_depth),
        "side_vert": T(side_vert_depth),
        "top_chord": T(top_depth),
        "bottom_chord": T(bottom_depth),
    }

    length = T(length)

    height = T(height)

    angle = T(angle)

    phi = torch.atan(((length - 2 * height / torch.tan(angle)) / 2) / height)

    phi = torch.atan(((length - 2 * height / torch.tan(angle)) / 2) / height)

    L_incline = height / torch.sin(angle)

    L_diagonal = height / torch.cos(phi)

    L_top = length - 2 * height / torch.tan(angle)

    L_bottom = length

    L_vert = height - thickness_dict["top_chord"] - thickness_dict["bottom_chord"]

    L_mid_vert = L_vert

    L_side_vert = L_vert

    volume = (
        2 * thickness_dict["incline"] * depth_dict["incline"] * L_incline
        + 2 * thickness_dict["diagonal"] * depth_dict["diagonal"] * L_diagonal
        + thickness_dict["top_chord"] * depth_dict["top_chord"] * L_top
        + thickness_dict["bottom_chord"] * depth_dict["bottom_chord"] * L_bottom
        + thickness_dict["mid_vert"] * depth_dict["mid_vert"] * L_mid_vert
        + 2 * thickness_dict["side_vert"] * depth_dict["side_vert"] * L_side_vert
    )

    return volume


def compute_weight_torch(
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
    length,
    height,
    angle,
    density=7850.0,
    dtype=DTYPE,
    device=None,
):
    """Compute weight (N) of OnePanelPratt2D truss (differentiable). Main members only."""

    def T(x):

        if isinstance(x, torch.Tensor):

            return x.to(dtype=dtype, device=device)

        return torch.tensor(x, dtype=dtype, device=device)

    volume = compute_volume_torch(
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
        length,
        height,
        angle,
        dtype=dtype,
        device=device,
    )

    density = T(density)

    g = T(9.81)

    weight = volume * density * g

    return weight


def compute_weight_with_laterals_torch(
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
    length,
    height,
    angle,
    density=7850.0,
    lateral_spacing=0.1524,
    epoxy_cost=0.515,
    dtype=DTYPE,
    device=None,
):
    """Compute weight including lateral bracing and epoxy (differentiable)."""

    def T(x):

        if isinstance(x, torch.Tensor):

            return x.to(dtype=dtype, device=device)

        return torch.tensor(x, dtype=dtype, device=device)

    main_weight = compute_weight_torch(
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
        length,
        height,
        angle,
        density=density,
        dtype=dtype,
        device=device,
    )

    length = T(length)

    height = T(height)

    angle = T(angle)

    top_depth = T(top_depth)

    bottom_depth = T(bottom_depth)

    lateral_spacing = T(lateral_spacing)

    epoxy_cost = T(epoxy_cost)

    lateral_thick = T(3.0 / 16.0 * 0.0254)

    t_lat_length = lateral_spacing - 2 * top_depth

    t_mid_length = torch.sqrt(
        (height / torch.tan(angle)) ** 2 + (lateral_spacing - 2 * top_depth) ** 2
    )

    b_mid_length = torch.sqrt(
        (height / torch.tan(angle)) ** 2 + (lateral_spacing - 2 * bottom_depth) ** 2
    )

    b_lat_length = lateral_spacing - 2 * bottom_depth

    b_out_length = torch.sqrt(
        (length - 2 * height / torch.tan(angle)) ** 2 + (lateral_spacing - 2 * bottom_depth) ** 2
    )

    lateral_depth = T(3.0 / 16.0 * 0.0254)

    lateral_volume = (
        3 * (lateral_thick * t_lat_length * lateral_depth)
        + 2 * (lateral_thick * t_mid_length * lateral_depth)
        + 2 * (lateral_thick * b_mid_length * lateral_depth)
        + 3 * (lateral_thick * b_lat_length * lateral_depth)
        + 2 * (lateral_thick * b_out_length * lateral_depth)
    )

    density = T(density)

    g = T(9.81)

    lateral_weight = (lateral_volume * density * g) / T(2.0)

    total_weight = main_weight + lateral_weight + epoxy_cost

    return total_weight
