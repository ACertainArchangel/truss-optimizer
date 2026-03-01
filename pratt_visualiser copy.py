from bridges_parametric import OnePanelPratt2D
import turtle
from math import sin, cos, tan
import numpy as np
from PIL import Image

class PrattVisualiser:

    @staticmethod
    def visualise(bridge: OnePanelPratt2D) -> np.ndarray:
        # Reset turtle's internal state to avoid interference from previous runs
        turtle.TurtleScreen._RUNNING = True
        turtle._Screen._root = None
        turtle._Screen._canvas = None


        try:
            screen = turtle.getscreen()
            screen.clearscreen()
        except turtle.TurtleGraphicsError:
            screen = turtle.Screen()
        screen.title("Pratt Bridge Visualiser")
        screen.setup(width=800, height=600)
        root = None  # No root to clean up for visible canvas
        

        # Set world coordinates
        screen.setworldcoordinates(-1, -1, bridge.length + 1, bridge.length + 1)
        
        # Create turtle and draw the bridge
        t = turtle.RawTurtle(screen)
        t.speed(0)
        t.hideturtle()
        t.pensize(2)
        t.color("black")
        t.penup()
        t.goto(0, 0)
        t.pendown()

        # Drawing commands (unchanged)
        t.forward(bridge.length)
        t.left(180 - bridge.angle * 180 / 3.14159)
        t.forward(bridge.height/sin(bridge.angle))
        t.setheading(180)
        t.forward(bridge.length-2*bridge.height/tan(bridge.angle))
        t.left(bridge.angle * 180 / 3.14159)
        t.forward(bridge.height/sin(bridge.angle))
        t.penup()
        t.backward(bridge.height/sin(bridge.angle))
        t.setheading(270)
        t.pendown()
        t.forward(bridge.height)
        t.penup()
        t.backward(bridge.height)
        t.pendown()
        t.left(bridge.phi * 180 / 3.14159)
        t.forward(bridge.height/cos(bridge.phi))
        t.setheading(90)
        t.forward(bridge.height)
        t.penup()
        t.backward(bridge.height)
        t.pendown()
        t.setheading(0)
        t.left(90 - bridge.phi * 180 / 3.14159)
        t.forward(bridge.height/cos(bridge.phi))
        t.setheading(270)
        t.forward(bridge.height)

        # Capture drawing as image
        canvas = screen.getcanvas()
        canvas.postscript(file="turtle_output.ps")
        img = Image.open("turtle_output.ps")
        img_gray = img.convert("L")
        arr = np.array(img_gray)

        # Reset turtle's internal state to ensure clean subsequent runs
        turtle.TurtleScreen._RUNNING = True
        turtle._Screen._root = None
        turtle._Screen._canvas = None

        return arr

    def __init__(self):
        pass

    def shutdown(self):
        # Cleanup method remains for external use if needed
        try:
            turtle.bye()
        except:
            pass
        turtle.TurtleScreen._RUNNING = True
        turtle._Screen._root = None
        turtle._Screen._canvas = None

if __name__ == "__main__":
    b = OnePanelPratt2D(
        angle=3.14159/6,
        height=2,
        length=10,
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
    vis = PrattVisualiser()