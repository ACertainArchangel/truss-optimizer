import math

import torch
from torch import atan, cos, sin, tan

from .base import BaseTruss
from .members import CompressionMember, TensionMember


class PrattTrussV1(BaseTruss):
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
        super().__init__(
            material, length, constraints, parameters, k, softmin_temperature, fixed_cost
        )

    @property
    def required_parameters(self):
        return {
            "compression_member_thickness": [
                "top_chord_thickness",
                "incline_thickness",
                "side_vertical_thickness",
                "mid_vertical_thickness",
            ],
            "compression_member_depth": [
                "top_chord_depth",
                "incline_depth",
                "side_vertical_depth",
                "mid_vertical_depth",
            ],
            "tension_member_thickness": ["bottom_chord_thickness", "diagonal_thickness"],
            "tension_member_depth": ["bottom_chord_depth", "diagonal_depth"],
            "misc": ["height", "angle"],
        }

    def visualize(self, keep_animation_up=False):
        """Draw the truss geometry using Python's built-in turtle module."""
        import math
        import turtle

        # Quick access
        angle = self.parameters["angle"].item()
        height = self.parameters["height"].item()
        x_incline = height / math.tan(angle)
        phi = math.atan((self.length / 2 - x_incline) / height)

        # reset for rerun safefy
        turtle.TurtleScreen._RUNNING = True
        turtle._Screen._root = None
        turtle._Screen._canvas = None

        # make another scree or clear it
        try:
            screen = turtle.getscreen()
            screen.clearscreen()
        except turtle.TurtleGraphicsError:
            screen = turtle.Screen()

        # title
        screen.title(
            f"Optimised Pratt Truss  |  Load/Weight = {self.load_to_weight_ratio():.1f}  |  "
            f"Critical Load = {self.critical_load():.1f} N"
        )

        # Draw
        screen.setup(width=800, height=600)
        margin = self.length * 0.15
        screen.setworldcoordinates(-margin, -margin * 2, self.length + margin, height * 2.5)
        t = turtle.RawTurtle(screen)
        t.speed(0)
        t.hideturtle()
        t.pensize(2)
        t.penup()
        t.goto(0, 0)
        t.pendown()
        t.forward(self.length)
        t.left(180 - math.degrees(angle))
        t.forward(height / math.sin(angle))
        t.setheading(180)
        t.forward(self.length - 2 * x_incline)
        t.left(math.degrees(angle))
        t.forward(height / math.sin(angle))
        t.penup()
        t.backward(height / math.sin(angle))
        t.setheading(270)
        t.pendown()
        t.forward(height)
        t.penup()
        t.backward(height)
        t.pendown()
        t.left(math.degrees(phi))
        t.forward(height / math.cos(phi))
        t.setheading(90)
        t.forward(height)
        t.penup()
        t.backward(height)
        t.pendown()
        t.setheading(0)
        t.left(90 - math.degrees(phi))
        t.forward(height / math.cos(phi))
        t.setheading(270)
        t.forward(height)
        screen.update()

        # Save with aweful ghost script
        import os
        import tempfile
        from io import BytesIO

        from PIL import Image

        canvas = screen.getcanvas()
        with tempfile.NamedTemporaryFile(suffix=".ps", delete=False) as tmp:
            tmp_path = tmp.name
            canvas.postscript(file=tmp_path, colormode="color")
        img = Image.open(tmp_path)
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        os.unlink(tmp_path)

        # Delete or keep
        if keep_animation_up:
            screen.mainloop()
        else:
            try:
                turtle.bye()
            except Exception:
                pass
            turtle.TurtleScreen._RUNNING = True
            turtle._Screen._root = None
            turtle._Screen._canvas = None

        return buf

    def generate_members(self):
        phi = atan(
            (self.length - 2 * (self.parameters["height"] / tan(self.parameters["angle"])))
            / (2 * self.parameters["height"])
        )

        members = [
            # Bottom chord
            TensionMember(
                length=torch.tensor(self.length, requires_grad=False),
                thickness=self.parameters["bottom_chord_thickness"],
                depth=self.parameters["bottom_chord_depth"],
                material=self.material,
                a=cos(self.parameters["angle"]) / (2 * sin(self.parameters["angle"]))
                + sin(phi) / (6 * cos(phi)),
                b=-(
                    self.parameters["side_vertical_thickness"]
                    * self.parameters["side_vertical_depth"]
                )
                / (
                    2
                    * self.parameters["side_vertical_thickness"]
                    * self.parameters["side_vertical_depth"]
                    + self.parameters["mid_vertical_thickness"]
                    * self.parameters["mid_vertical_depth"]
                )
                * (1 / 3)
                * (self.length / 2 - self.parameters["height"] / tan(self.parameters["angle"])),
                k=self.k,
                softmin_temperature=self.softmin_temperature,
            ),
            # Top chord
            CompressionMember(
                length=self.length
                - 2 * self.parameters["height"] / torch.tan(self.parameters["angle"]),
                thickness=self.parameters["top_chord_thickness"],
                depth=self.parameters["top_chord_depth"],
                material=self.material,
                a=cos(self.parameters["angle"]) / (2 * sin(self.parameters["angle"]))
                + sin(phi) / (6 * cos(phi)),
                b=(
                    self.parameters["side_vertical_thickness"]
                    * self.parameters["side_vertical_depth"]
                )
                / (
                    2
                    * self.parameters["side_vertical_thickness"]
                    * self.parameters["side_vertical_depth"]
                    + self.parameters["mid_vertical_thickness"]
                    * self.parameters["mid_vertical_depth"]
                )
                * (1 / 3)
                * (self.length / 2 - self.parameters["height"] / tan(self.parameters["angle"])),
                k=self.k,
                softmin_temperature=self.softmin_temperature,
            ),
            # Diagonals
            TensionMember(
                length=self.parameters["height"] / cos(phi),
                thickness=self.parameters["diagonal_thickness"],
                depth=self.parameters["diagonal_depth"],
                material=self.material,
                a=1 / (6 * cos(phi)),
                b=torch.tensor(0.0),
                k=self.k,
                softmin_temperature=self.softmin_temperature,
            ),
            # Inclines
            CompressionMember(
                length=self.parameters["height"] / sin(self.parameters["angle"]),
                thickness=self.parameters["incline_thickness"],
                depth=self.parameters["incline_depth"],
                material=self.material,
                a=1 / (2 * sin(self.parameters["angle"])),
                b=torch.tensor(0.0),
                k=self.k,
                softmin_temperature=self.softmin_temperature,
            ),
            # Side verticals
            CompressionMember(
                length=self.parameters["height"],
                thickness=self.parameters["side_vertical_thickness"],
                depth=self.parameters["side_vertical_depth"],
                material=self.material,
                a=(
                    self.parameters["side_vertical_thickness"]
                    * self.parameters["side_vertical_depth"]
                )
                / (
                    2
                    * self.parameters["side_vertical_thickness"]
                    * self.parameters["side_vertical_depth"]
                    + self.parameters["mid_vertical_thickness"]
                    * self.parameters["mid_vertical_depth"]
                )
                * (1 / 3),
                b=torch.tensor(0.0),
                k=self.k,
                softmin_temperature=self.softmin_temperature,
            ),
            # Mid verticals
            CompressionMember(
                length=self.parameters["height"],
                thickness=self.parameters["mid_vertical_thickness"],
                depth=self.parameters["mid_vertical_depth"],
                material=self.material,
                a=(
                    self.parameters["mid_vertical_thickness"]
                    * self.parameters["mid_vertical_depth"]
                )
                / (
                    2
                    * self.parameters["side_vertical_thickness"]
                    * self.parameters["side_vertical_depth"]
                    + self.parameters["mid_vertical_thickness"]
                    * self.parameters["mid_vertical_depth"]
                )
                * (1 / 3),
                b=torch.tensor(0.0),
                k=self.k,
                softmin_temperature=self.softmin_temperature,
            ),
        ]

        return members

    def generate_report(self):
        return (
            "Write something here lol."
            + str(self.parameters["angle"].item())
            + str(self.parameters["height"])
        )

    def clamp_params_to_valid_ranges(self, min_top_size_in_meters=0):
        with torch.no_grad():  # Because this is none of the computation graph's business
            self.parameters["angle"].clamp_(
                min=torch.atan(
                    2 * self.parameters["height"] / (self.length - min_top_size_in_meters)
                )
                + 1e-3
            )
