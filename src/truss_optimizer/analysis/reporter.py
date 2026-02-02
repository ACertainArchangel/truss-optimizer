"""
Failure mode analysis and reporting.

This module provides tools for analyzing bridge failure modes and
generating detailed structural reports.
"""

from __future__ import annotations

from typing import Dict, Optional, Union
from pathlib import Path

from truss_optimizer.core.truss import PrattTruss
from truss_optimizer.materials import Material


class FailureAnalyzer:
    """
    Analyzer for bridge failure modes with detailed reporting.
    
    Provides comprehensive analysis of all failure modes and generates
    reports suitable for engineering documentation.
    
    Args:
        bridge: PrattTruss to analyze
        
    Example:
        >>> from truss_optimizer import PrattTruss, FailureAnalyzer, materials
        >>> 
        >>> bridge = PrattTruss(span=0.47, height=0.15, material=materials.BalsaWood())
        >>> analyzer = FailureAnalyzer(bridge)
        >>> analyzer.print_report()
    """
    
    def __init__(self, bridge: PrattTruss):
        self.bridge = bridge
        self._failure_modes: Optional[Dict[str, float]] = None
    
    @property
    def failure_modes(self) -> Dict[str, float]:
        """Get all failure mode critical loads."""
        if self._failure_modes is None:
            self._failure_modes = self.bridge.get_failure_modes()
        return self._failure_modes
    
    @property
    def critical_load(self) -> float:
        """Critical (governing) failure load."""
        return self.bridge.critical_load
    
    @property
    def governing_mode(self) -> str:
        """Name of the governing failure mode."""
        return self.bridge.governing_failure_mode
    
    @property
    def safety_factors(self) -> Dict[str, float]:
        """
        Safety factors for each failure mode.
        
        Safety factor = failure_load / governing_load
        A value > 1 means the mode has margin above the governing mode.
        """
        governing = self.critical_load
        return {
            mode: load / governing if governing > 0 else float('inf')
            for mode, load in self.failure_modes.items()
            if load is not None
        }
    
    def get_member_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get failure summary grouped by member.
        
        Returns:
            Dict mapping member names to their failure mode loads
        """
        summary: Dict[str, Dict[str, float]] = {}
        
        for mode, load in self.failure_modes.items():
            if load is None:
                continue
            
            # Extract member name from mode
            parts = mode.rsplit('_', 1)
            if len(parts) == 2 and parts[1] in ('rupture', 'buckle', 'stress'):
                member = parts[0]
            elif '_buckle_out_of_plane' in mode:
                member = mode.replace('_buckle_out_of_plane', '')
            elif '_combined_stress' in mode:
                member = mode.replace('_combined_stress', '')
            elif '_buckle' in mode:
                member = mode.replace('_buckle', '')
            elif '_rupture' in mode:
                member = mode.replace('_rupture', '')
            else:
                member = mode
            
            if member not in summary:
                summary[member] = {}
            summary[member][mode] = load
        
        return summary
    
    def format_report(self, unit_system: str = 'metric') -> str:
        """
        Generate a formatted analysis report.
        
        Args:
            unit_system: 'metric' (m, N) or 'imperial' (in, lbf)
            
        Returns:
            Formatted report string
        """
        # Unit conversion factors
        if unit_system == 'imperial':
            length_factor = 39.3701  # m to inches
            force_factor = 0.2248  # N to lbf
            length_unit = 'in'
            force_unit = 'lbf'
        else:
            length_factor = 100  # m to cm
            force_factor = 1.0
            length_unit = 'cm'
            force_unit = 'N'
        
        lines = [
            "",
            "=" * 70,
            "  STRUCTURAL FAILURE ANALYSIS REPORT",
            "=" * 70,
            "",
            "  BRIDGE GEOMETRY",
            "  " + "-" * 40,
            f"  Span:      {self.bridge.span * length_factor:>10.2f} {length_unit}",
            f"  Height:    {self.bridge.height * length_factor:>10.2f} {length_unit}",
            f"  Angle:     {self.bridge.angle * 180 / 3.14159:>10.1f}°",
            "",
        ]
        
        if self.bridge.material:
            lines.extend([
                "  MATERIAL PROPERTIES",
                "  " + "-" * 40,
                f"  Material:  {self.bridge.material}",
                f"  E:         {self.bridge.E / 1e9:>10.2f} GPa",
                f"  σ_comp:    {self.bridge.sigma_compression / 1e6:>10.2f} MPa",
                f"  σ_tens:    {self.bridge.sigma_tension / 1e6:>10.2f} MPa",
                f"  Density:   {self.bridge.density:>10.1f} kg/m³",
                "",
            ])
        
        lines.extend([
            "  FAILURE MODE ANALYSIS",
            "  " + "-" * 40,
            f"  {'Mode':<35} {'Load':>10} {'SF':>8}",
            "  " + "-" * 55,
        ])
        
        # Sort by critical load
        sorted_modes = sorted(
            self.failure_modes.items(),
            key=lambda x: x[1] if x[1] else float('inf')
        )
        
        for mode, load in sorted_modes:
            if load is None:
                continue
            sf = self.safety_factors.get(mode, float('inf'))
            load_display = f"{load * force_factor:.2f}" if load < 1e6 else f"{load:.2e}"
            sf_display = f"{sf:.2f}" if sf < 100 else ">99"
            marker = " ◀ GOVERNING" if mode == self.governing_mode else ""
            lines.append(f"  {mode:<35} {load_display:>10} {sf_display:>8}{marker}")
        
        lines.extend([
            "",
            "  SUMMARY",
            "  " + "-" * 40,
            f"  Governing Mode:  {self.governing_mode}",
            f"  Critical Load:   {self.critical_load * force_factor:.2f} {force_unit}",
            f"  Weight:          {self.bridge.weight * force_factor:.4f} {force_unit}",
            f"  Load/Weight:     {self.bridge.load_to_weight:.1f}",
            "",
            "=" * 70,
            "",
        ])
        
        return "\n".join(lines)
    
    def print_report(self, unit_system: str = 'metric') -> None:
        """Print the analysis report to console."""
        print(self.format_report(unit_system))
    
    def save_report(
        self, 
        path: Union[str, Path], 
        unit_system: str = 'metric'
    ) -> None:
        """
        Save the analysis report to a file.
        
        Args:
            path: Output file path
            unit_system: 'metric' or 'imperial'
        """
        path = Path(path)
        with open(path, 'w') as f:
            f.write(self.format_report(unit_system))


def analyze_bridge(
    bridge: PrattTruss,
    print_report: bool = True,
    unit_system: str = 'metric',
) -> FailureAnalyzer:
    """
    Convenience function to analyze a bridge and optionally print report.
    
    Args:
        bridge: Bridge to analyze
        print_report: Whether to print the report
        unit_system: 'metric' or 'imperial'
        
    Returns:
        FailureAnalyzer instance
        
    Example:
        >>> from truss_optimizer import PrattTruss, materials
        >>> from truss_optimizer.analysis import analyze_bridge
        >>> 
        >>> bridge = PrattTruss(span=0.47, material=materials.BalsaWood())
        >>> analyzer = analyze_bridge(bridge)
    """
    analyzer = FailureAnalyzer(bridge)
    
    if print_report:
        analyzer.print_report(unit_system)
    
    return analyzer
