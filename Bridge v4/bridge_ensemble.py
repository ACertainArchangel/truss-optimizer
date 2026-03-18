"""
A simplified version of the workflow I actually used for the bridge project.
By simplified I don't mean it acts more simply or less completely, I just mean
it has less garbage code and lives in one file instead of 1000000 and a bash script.
:p
"""

fixing_angle_and_height_in_second_round = True

number_of_bridges_per_type = 10


import math

import torch
from fpdf import FPDF
from src.truss_optimizer.materials import BalsaWood, Epoxy, rule_of_mixtures
from src.truss_optimizer.optimizer import Optimizer  # TODO: remove the .src later
from src.truss_optimizer.trusses import PrattTrussV1, PrattTrussV2
from src.truss_optimizer.utils import fig_to_image, inches_to_meters, meters_to_inches

balsa_laminate = rule_of_mixtures(
    BalsaWood(), Epoxy(), 1, "Balsa Laminate", ideality_coefficient=0.90
)  # 50% balsa, 50% epoxy by volume. Generally, laminates lose 5-15% effeciency so taking the average and setting eta=0.9 is a good rule of thumb.

constraints = {
    "height": (inches_to_meters(4.0), inches_to_meters(12.0)),
    "angle": (math.radians(30), math.radians(50)),
    "compression_member_thickness": (inches_to_meters(0.125), inches_to_meters(0.5)),
    "tension_member_thickness": (inches_to_meters(0.125), inches_to_meters(0.5)),
    "compression_member_depth": (inches_to_meters(0.125), inches_to_meters(0.5)),
    "tension_member_depth": (inches_to_meters(0.125), inches_to_meters(0.5)),
}

v1 = [
    PrattTrussV1(
        material=balsa_laminate, length=inches_to_meters(18.5), constraints=constraints, k=0.5
    )
    for _ in range(number_of_bridges_per_type)
]

v2 = [
    PrattTrussV2(
        material=balsa_laminate, length=inches_to_meters(18.5), constraints=constraints, k=0.5
    )
    for _ in range(number_of_bridges_per_type)
]

v1_optimizers = [
    Optimizer(
        bridge=v1bridge,
        objective="load_to_weight",
        learning_rate=0.001,
        unit_system="metric",
        constraints=constraints,
        fixed_parameters=["length"],
    )
    for v1bridge in v1
]

v2_optimizers = [
    Optimizer(
        bridge=v2bridge,
        objective="load_to_weight",
        learning_rate=0.001,
        unit_system="metric",
        constraints=constraints,
        fixed_parameters=["length"],
    )
    for v2bridge in v2
]

v1_plots = [optimizer.optimize(iterations=300, verbose=True) for optimizer in v1_optimizers]
v2_plots = [optimizer.optimize(iterations=300, verbose=True) for optimizer in v2_optimizers]

v1_bridges = [optimizer.bridge for optimizer in v1_optimizers]
v2_bridges = [optimizer.bridge for optimizer in v2_optimizers]

v1_bridges_rounded = [
    bridge.rounded(
        numerator=1, denominator=16, rounding=["compression_member_depth", "tension_member_depth"]
    )
    for bridge in v1_bridges
]
v2_bridges_rounded = [
    bridge.rounded(
        numerator=1, denominator=16, rounding=["compression_member_depth", "tension_member_depth"]
    )
    for bridge in v2_bridges
]

if fixing_angle_and_height_in_second_round:
    for bridge in v1_bridges_rounded + v2_bridges_rounded:
        bridge.clamp_params_to_valid_ranges(
            min_top_size_in_meters=0.1524
        )  # TODO: THIS IS AN EXCEPTIONALLY BAD HACK AND THIS NEEDS TO BE MADE BETTER!!! THIS ENTIRE BLOCK NEEDS TO BE ONE METHOD AND WORK FROM CONSTRAINTS.
        with torch.no_grad():
            for name, param in bridge.parameters.items():
                if name in constraints:
                    lo, hi = constraints[name]

                    param.clamp_(lo + 1e-3, hi - 1e-3)
        bridge.clamp_params_to_valid_ranges(
            min_top_size_in_meters=0.1524
        )  # TODO: THIS IS AN EXCEPTIONALLY BAD HACK AND THIS NEEDS TO BE MADE BETTER!!! THIS ENTIRE BLOCK NEEDS TO BE ONE METHOD AND WORK FROM CONSTRAINTS.

v1_second_round_optimizers = [
    Optimizer(
        bridge=bridge,
        objective="load_to_weight",
        learning_rate=0.0001,
        unit_system="metric",
        constraints=constraints,
        fixed_parameters=["length", "compression_member_depth", "tension_member_depth"]
        + (["angle", "height"] if fixing_angle_and_height_in_second_round else []),
    )
    for bridge in v1_bridges_rounded
]

v2_second_round_optimizers = [
    Optimizer(
        bridge=bridge,
        objective="load_to_weight",
        learning_rate=0.0001,
        unit_system="metric",
        constraints=constraints,
        fixed_parameters=["length", "compression_member_depth", "tension_member_depth"]
        + (["angle", "height"] if fixing_angle_and_height_in_second_round else []),
    )
    for bridge in v2_bridges_rounded
]

v1_second_round_plots = [
    optimizer.optimize(iterations=300, verbose=True) for optimizer in v1_second_round_optimizers
]
v2_second_round_plots = [
    optimizer.optimize(iterations=300, verbose=True) for optimizer in v2_second_round_optimizers
]

v1_bridges_final = [optimizer.bridge for optimizer in v1_second_round_optimizers]
v2_bridges_final = [optimizer.bridge for optimizer in v2_second_round_optimizers]

v1_best_idx = max(
    range(len(v1_bridges_final)), key=lambda i: v1_bridges_final[i].load_to_weight_ratio()
)
v2_best_idx = max(
    range(len(v2_bridges_final)), key=lambda i: v2_bridges_final[i].load_to_weight_ratio()
)

v1_best_bridge = v1_bridges_final[v1_best_idx]
v2_best_bridge = v2_bridges_final[v2_best_idx]

v1_best_plot = fig_to_image(v1_plots[v1_best_idx])
v2_best_plot = fig_to_image(v2_plots[v2_best_idx])

v1_second_round_best_plot = fig_to_image(
    v1_second_round_plots[v1_bridges_final.index(v1_best_bridge)]
)
v2_second_round_best_plot = fig_to_image(
    v2_second_round_plots[v2_bridges_final.index(v2_best_bridge)]
)
v1_best_report = v1_best_bridge.generate_report()
v2_best_report = v2_best_bridge.generate_report()

v1_pdf = FPDF()
v1_pdf.set_auto_page_break(auto=True, margin=15)
v1_pdf.add_page()
v1_pdf.set_font("Times", size=12)
v1_pdf.multi_cell(w=0, h=10, text=v1_best_report, new_x="LMARGIN", new_y="NEXT")
v1_pdf.ln(5)
v1_pdf.image(v1_best_plot, x=10, w=180)
v1_pdf.multi_cell(
    w=0,
    h=8,
    text="Figure 1: Convergence plot for the first round of optimization.",
    new_x="LMARGIN",
    new_y="NEXT",
)
v1_pdf.ln(5)
v1_pdf.image(v1_second_round_best_plot, x=10, w=180)
v1_pdf.multi_cell(
    w=0,
    h=8,
    text="Figure 2: Convergence plot for the second round of optimization after rounding and fixing parameters.",
    new_x="LMARGIN",
    new_y="NEXT",
)
v1_pdf.ln(5)
v1_pdf.image(v1_best_bridge.visualize(), x=10, w=180)
v1_pdf.multi_cell(
    w=0,
    h=8,
    text="Figure 3: Visualization of the best PrattTrussV1 bridge design after optimization.",
    new_x="LMARGIN",
    new_y="NEXT",
)
v1_pdf.ln(5)
v1_pdf.output("v1_best_bridge_report.pdf")

v2_pdf = FPDF()
v2_pdf.set_auto_page_break(auto=True, margin=15)
v2_pdf.add_page()
v2_pdf.set_font("Times", size=12)
v2_pdf.multi_cell(w=0, h=10, text=v2_best_report, new_x="LMARGIN", new_y="NEXT")
v2_pdf.ln(5)
v2_pdf.image(v2_best_plot, x=10, w=180)
v2_pdf.multi_cell(
    w=0,
    h=8,
    text="Figure 1: Convergence plot for the first round of optimization.",
    new_x="LMARGIN",
    new_y="NEXT",
)
v2_pdf.ln(5)
v2_pdf.image(v2_second_round_best_plot, x=10, w=180)
v2_pdf.multi_cell(
    w=0,
    h=8,
    text="Figure 2: Convergence plot for the second round of optimization after rounding and fixing parameters.",
    new_x="LMARGIN",
    new_y="NEXT",
)
v2_pdf.ln(5)
v2_pdf.image(v2_best_bridge.visualize(), x=10, w=180)
v2_pdf.multi_cell(
    0, 8, "Figure 3: Visualization of the best PrattTrussV2 bridge design after optimization."
)
v2_pdf.output("v2_best_bridge_report.pdf")
