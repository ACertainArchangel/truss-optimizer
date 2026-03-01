"""
Optimization result container with visualization and export capabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class OptimizationResult:
    """
    Container for optimization results with analysis and export methods.
    
    Attributes:
        params: Final optimized parameters
        initial_params: Initial parameters before optimization
        critical_load: Maximum applied load before failure (N)
        weight: Total truss weight (N)
        load_to_weight: Load-to-weight ratio
        governing_mode: Name of the governing failure mode
        failure_modes: Dict of all failure mode critical loads
        history: Optimization history (loss, params over iterations)
        iterations: Number of optimization iterations
        final_loss: Final loss value
        best_iteration: Iteration where best result was found
        
    Example:
        >>> result = optimizer.optimize(iterations=5000)
        >>> print(f"Load/Weight: {result.load_to_weight:.1f}")
        >>> result.visualize()
        >>> result.export("my_bridge.json")
    """
    
    params: Dict[str, float]
    initial_params: Dict[str, float]
    critical_load: float
    weight: float
    load_to_weight: float
    governing_mode: str
    failure_modes: Dict[str, float]
    history: Dict[str, List[float]] = field(default_factory=dict)
    iterations: int = 0
    final_loss: float = 0.0
    best_iteration: int = 0
    
    @property
    def improvement(self) -> float:
        """Improvement in objective from initial to final (%)."""
        if not self.history.get('loss'):
            return 0.0
        initial = self.history['loss'][0]
        final = self.final_loss
        if initial == 0:
            return 0.0
        return abs((final - initial) / initial * 100)
    
    def summary(self) -> str:
        """Generate a human-readable summary of the optimization result."""
        lines = [
            "=" * 60,
            "  OPTIMIZATION RESULT",
            "=" * 60,
            "",
            f"  Critical Load:      {self.critical_load:>10.2f} N",
            f"  Weight:             {self.weight:>10.4f} N",
            f"  Load/Weight Ratio:  {self.load_to_weight:>10.1f}",
            "",
            f"  Governing Mode:     {self.governing_mode}",
            f"  Iterations:         {self.iterations}",
            f"  Best at Iteration:  {self.best_iteration}",
            "",
            "  Top 5 Failure Modes:",
        ]
        
        # Sort failure modes by critical load
        sorted_modes = sorted(
            self.failure_modes.items(), 
            key=lambda x: x[1] if x[1] else float('inf')
        )
        for mode, load in sorted_modes[:5]:
            lines.append(f"    {mode:.<35} {load:>10.2f} N")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def print_summary(self) -> None:
        """Print optimization summary to console."""
        print(self.summary())
    
    def export(self, path: Union[str, Path], format: str = "json") -> None:
        """
        Export optimization result to file.
        
        Args:
            path: Output file path
            format: Export format ("json" or "dxf")
            
        Example:
            >>> result.export("bridge.json")
            >>> result.export("bridge.dxf")  # For CAD import
        """
        path = Path(path)
        
        if format == "json" or path.suffix == ".json":
            self._export_json(path)
        elif format == "dxf" or path.suffix == ".dxf":
            self._export_dxf(path)
        else:
            raise ValueError(f"Unknown export format: {format}")
    
    def _export_json(self, path: Path) -> None:
        """Export to JSON format."""
        data = {
            "parameters": self.params,
            "initial_parameters": self.initial_params,
            "results": {
                "critical_load_N": self.critical_load,
                "weight_N": self.weight,
                "load_to_weight_ratio": self.load_to_weight,
                "governing_failure_mode": self.governing_mode,
            },
            "failure_modes": self.failure_modes,
            "optimization": {
                "iterations": self.iterations,
                "best_iteration": self.best_iteration,
                "final_loss": self.final_loss,
            },
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _export_dxf(self, path: Path) -> None:
        """Export geometry to DXF format for CAD software."""
        # This would require ezdxf or similar library
        raise NotImplementedError(
            "DXF export requires the 'ezdxf' package. "
            "Install with: pip install ezdxf"
        )
    

    def visualize(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """Draw the truss geometry using Python's built-in turtle module."""
        import turtle
        import math

        p = self.params
        angle = p.get('angle', 0.52)
        height = p.get('height', 0.15)
        span = p.get('span', p.get('length', 0.47))

        x_incline = height / math.tan(angle)
        phi = math.atan((span / 2 - x_incline) / height)

        # Reset turtle state for clean re-runs
        turtle.TurtleScreen._RUNNING = True
        turtle._Screen._root = None
        turtle._Screen._canvas = None

        try:
            screen = turtle.getscreen()
            screen.clearscreen()
        except turtle.TurtleGraphicsError:
            screen = turtle.Screen()

        screen.title(
            f"Optimised Pratt Truss  |  Load/Weight = {self.load_to_weight:.1f}  |  "
            f"Critical Load = {self.critical_load:.1f} N"
        )
        screen.setup(width=800, height=600)

        margin = span * 0.15
        screen.setworldcoordinates(-margin, -margin * 2, span + margin, height * 2.5)

        t = turtle.RawTurtle(screen)
        t.speed(0)
        t.hideturtle()
        t.pensize(2)
        t.penup()
        t.goto(0, 0)
        t.pendown()
        t.forward(span)
        t.left(180 - math.degrees(angle))
        t.forward(height / math.sin(angle))
        t.setheading(180)
        t.forward(span - 2 * x_incline)
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

        if save_path:
            canvas = screen.getcanvas()
            ps_path = str(save_path).rsplit('.', 1)[0] + '.ps'
            canvas.postscript(file=ps_path)
            try:
                from PIL import Image
                img = Image.open(ps_path)
                img.save(save_path)
            except ImportError:
                pass  # PostScript file saved at least

        if show:
            screen.mainloop()
        else:
            try:
                turtle.bye()
            except Exception:
                pass
            turtle.TurtleScreen._RUNNING = True
            turtle._Screen._root = None
            turtle._Screen._canvas = None
    
    def plot_convergence(self, show: bool = True, save_path: Optional[str] = None) -> Any:
        """
        Plot optimization convergence history.
        
        Args:
            show: Whether to display the plot
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Plotting requires matplotlib. "
                "Install with: pip install matplotlib"
            )
        
        if not self.history.get('loss'):
            raise ValueError("No optimization history available")

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        ax.plot(self.history['loss'], 'b-', linewidth=1)
        ax.axvline(self.best_iteration, color='r', linestyle='--',
                   label=f'Best (iter {self.best_iteration})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Optimization Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
