"""
Example 06: Creating a Custom Truss Type

This example shows how to create your own truss type by subclassing
BaseTruss and implementing differentiable physics.
"""

import math
import torch
from truss_optimizer import (
    BaseTruss, 
    FailureMode,
    BridgeOptimizer,
    softmin, 
    euler_buckling_load, 
    DTYPE,
    materials,
)


class SimpleTruss(BaseTruss):
    """
    A minimal truss with parallel chords and X-bracing.
    
    Geometry:
        ═══════════════════════════  (top chord - compression)
        ╲         ╱╲         ╱
         ╲       ╱  ╲       ╱
          ╲     ╱    ╲     ╱        (diagonals - tension)
           ╲   ╱      ╲   ╱
            ╲ ╱        ╲ ╱
        ═══════════════════════════  (bottom chord - tension)
        
    Load applied at center of bottom chord.
    """
    
    def __init__(
        self,
        span: float,
        height: float,
        material,
        chord_thickness: float = 0.01,
        chord_depth: float = 0.02,
        diagonal_thickness: float = 0.005,
        diagonal_depth: float = 0.01,
    ):
        self.span = span
        self.height = height
        self.material = material
        
        # Store as dict for easy access
        self._params = {
            'chord_thickness': chord_thickness,
            'chord_depth': chord_depth,
            'diagonal_thickness': diagonal_thickness,
            'diagonal_depth': diagonal_depth,
        }
        
        self._define_geometry()
    
    def _define_geometry(self):
        """Calculate derived geometric properties."""
        # Diagonal length (two X-braces, each spanning half the truss)
        half_span = self.span / 2
        self.diagonal_length = math.sqrt(half_span**2 + self.height**2)
        self.diagonal_angle = math.atan(self.height / half_span)
    
    def get_failure_modes(self):
        """
        Compute all failure modes and return as list.
        
        This truss has 4 failure modes:
        1. Bottom chord rupture (tension)
        2. Top chord buckling in-plane
        3. Top chord buckling out-of-plane  
        4. Top chord crushing (combined stress)
        """
        p = self._params
        E = self.material.E
        sigma_t = self.material.sigma_tension
        sigma_c = self.material.sigma_compression
        
        # Cross-section properties
        A_chord = p['chord_thickness'] * p['chord_depth']
        I_in = p['chord_thickness'] * p['chord_depth']**3 / 12  # About depth
        I_out = p['chord_depth'] * p['chord_thickness']**3 / 12  # About thickness
        
        modes = []
        
        # --- Failure Mode 1: Bottom chord tension rupture ---
        # Simple beam: max chord force ≈ M_max / height = (F*L/4) / h
        # So F_chord = F_applied * L / (4*h)
        # Rupture when: sigma_t * A = F_chord
        # F_applied = sigma_t * A * 4*h / L
        F_tension_rupture = sigma_t * A_chord * 4 * self.height / self.span
        modes.append(FailureMode(
            name='bottom_chord_rupture',
            member='bottom_chord',
            mechanism='tension',
            max_load=F_tension_rupture,
            description='Bottom chord yields in tension'
        ))
        
        # --- Failure Mode 2: Top chord buckling (in-plane) ---
        # Top chord has same force as bottom chord
        # Euler buckling: F_cr = π²EI/(KL)²
        # Using K=0.5 (fixed-fixed), effective length = full span
        K = 0.5
        F_euler_in = (math.pi**2 * E * I_in) / (K * self.span)**2
        force_coeff = self.span / (4 * self.height)
        F_buckle_in = F_euler_in / force_coeff
        modes.append(FailureMode(
            name='top_chord_buckle_in_plane',
            member='top_chord',
            mechanism='buckling',
            max_load=F_buckle_in,
            description='Top chord buckles in the plane of the truss'
        ))
        
        # --- Failure Mode 3: Top chord buckling (out-of-plane) ---
        F_euler_out = (math.pi**2 * E * I_out) / (K * self.span)**2
        F_buckle_out = F_euler_out / force_coeff
        modes.append(FailureMode(
            name='top_chord_buckle_out_of_plane',
            member='top_chord',
            mechanism='buckling',
            max_load=F_buckle_out,
            description='Top chord buckles out of the plane'
        ))
        
        # --- Failure Mode 4: Top chord crushing ---
        F_crush = sigma_c * A_chord / force_coeff
        modes.append(FailureMode(
            name='top_chord_crush',
            member='top_chord',
            mechanism='compression',
            max_load=F_crush,
            description='Top chord material crushes'
        ))
        
        return modes
    
    def compute_max_load_torch(self, **params):
        """
        Differentiable max load calculation.
        
        This is the KEY method for optimization - it must:
        1. Accept parameters as keyword arguments
        2. Use only torch operations (no numpy, no python min/max)
        3. Use softmin instead of min for differentiability
        4. Return (max_load_tensor, governing_mode_name)
        """
        # Merge provided params with defaults
        p = {**self._params, **params}
        
        # Convert to tensors for gradient flow
        chord_t = torch.as_tensor(p['chord_thickness'], dtype=DTYPE)
        chord_d = torch.as_tensor(p['chord_depth'], dtype=DTYPE)
        
        E = torch.as_tensor(self.material.E, dtype=DTYPE)
        sigma_t = torch.as_tensor(self.material.sigma_tension, dtype=DTYPE)
        sigma_c = torch.as_tensor(self.material.sigma_compression, dtype=DTYPE)
        
        span = torch.as_tensor(self.span, dtype=DTYPE)
        height = torch.as_tensor(self.height, dtype=DTYPE)
        
        # Cross-section properties (differentiable w.r.t. dimensions)
        A = chord_t * chord_d
        I_in = chord_t * chord_d**3 / 12
        I_out = chord_d * chord_t**3 / 12
        
        # Force coefficient: F_chord = F_applied * coeff
        force_coeff = span / (4 * height)
        
        # Failure mode 1: Tension rupture
        F_tension = sigma_t * A / force_coeff
        
        # Failure mode 2: In-plane buckling
        K = 0.5
        L_eff = K * span
        F_euler_in = euler_buckling_load(E, I_in, K=1.0, L=L_eff)  # K already in L_eff
        F_buckle_in = F_euler_in / force_coeff
        
        # Failure mode 3: Out-of-plane buckling
        F_euler_out = euler_buckling_load(E, I_out, K=1.0, L=L_eff)
        F_buckle_out = F_euler_out / force_coeff
        
        # Failure mode 4: Crushing
        F_crush = sigma_c * A / force_coeff
        
        # Stack all failure loads
        all_modes = torch.stack([F_tension, F_buckle_in, F_buckle_out, F_crush])
        
        # Use softmin for differentiable minimum
        max_load = softmin(all_modes, alpha=100.0)
        
        # Determine governing mode (for reporting, not optimization)
        mode_idx = torch.argmin(all_modes).item()
        mode_names = [
            'bottom_chord_rupture',
            'top_chord_buckle_in_plane', 
            'top_chord_buckle_out_of_plane',
            'top_chord_crush'
        ]
        
        return max_load, mode_names[mode_idx]
    
    def compute_volume_torch(self, **params):
        """
        Differentiable volume calculation.
        """
        p = {**self._params, **params}
        
        chord_t = torch.as_tensor(p['chord_thickness'], dtype=DTYPE)
        chord_d = torch.as_tensor(p['chord_depth'], dtype=DTYPE)
        diag_t = torch.as_tensor(p['diagonal_thickness'], dtype=DTYPE)
        diag_d = torch.as_tensor(p['diagonal_depth'], dtype=DTYPE)
        
        # Two chords (top and bottom)
        chord_volume = 2 * self.span * chord_t * chord_d
        
        # Four diagonals (2 X-braces with 2 members each)
        diagonal_volume = 4 * self.diagonal_length * diag_t * diag_d
        
        return chord_volume + diagonal_volume
    
    @property
    def critical_load(self) -> float:
        """Maximum load before first failure."""
        max_load, _ = self.compute_max_load_torch()
        return max_load.item()
    
    @property
    def volume(self) -> float:
        """Total material volume in m³."""
        return self.compute_volume_torch().item()
    
    @property
    def governing_failure_mode(self) -> str:
        """Name of the governing failure mode."""
        _, mode = self.compute_max_load_torch()
        return mode
    
    def get_optimizable_params(self):
        """Parameters that can be tuned by the optimizer."""
        return self._params.copy()
    
    def get_param_bounds(self):
        """Default bounds for parameters."""
        return {
            'chord_thickness': (0.002, 0.05),
            'chord_depth': (0.005, 0.10),
            'diagonal_thickness': (0.002, 0.03),
            'diagonal_depth': (0.005, 0.05),
        }
    
    def __repr__(self):
        return f"SimpleTruss(span={self.span:.3f}m, height={self.height:.3f}m)"


def main():
    print("=" * 60)
    print("CUSTOM TRUSS TYPE EXAMPLE")
    print("=" * 60)
    
    # Create our custom truss
    truss = SimpleTruss(
        span=0.5,           # 50 cm span
        height=0.1,         # 10 cm height
        material=materials.BalsaWood(),
        chord_thickness=0.006,
        chord_depth=0.012,
        diagonal_thickness=0.003,
        diagonal_depth=0.006,
    )
    
    print(f"\nInitial design: {truss}")
    print(f"  Critical load: {truss.critical_load:.2f} N")
    print(f"  Volume: {truss.volume * 1e6:.2f} mm³")
    print(f"  Weight: {truss.weight:.4f} kg")
    print(f"  Load/Weight: {truss.load_to_weight:.1f}")
    print(f"  Governing mode: {truss.governing_failure_mode}")
    
    # Show all failure modes
    print("\nFailure modes:")
    for mode in truss.get_failure_modes():
        print(f"  {mode.name}: {mode.max_load:.2f} N - {mode.description}")
    
    # Optimize it!
    print("\n" + "=" * 60)
    print("OPTIMIZING...")
    print("=" * 60)
    
    optimizer = BridgeOptimizer(
        bridge=truss,
        objective='load_to_weight',
        learning_rate=0.001,
        constraints={
            # Keep chord depth >= thickness (practical constraint)
            'chord_thickness': (0.003, 0.015),
            'chord_depth': (0.006, 0.025),
            'diagonal_thickness': (0.002, 0.010),
            'diagonal_depth': (0.004, 0.015),
        }
    )
    
    result = optimizer.optimize(iterations=3000, verbose=True)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\nOptimized parameters:")
    for name, value in result.parameters.items():
        if 'thickness' in name or 'depth' in name:
            print(f"  {name}: {value*1000:.2f} mm")
    
    print(f"\nPerformance:")
    print(f"  Critical load: {result.critical_load:.2f} N")
    print(f"  Weight: {result.weight:.4f} kg")
    print(f"  Load/Weight: {result.load_to_weight:.1f}")
    print(f"  Governing mode: {result.governing_mode}")
    
    # Calculate improvement
    improvement = result.load_to_weight / truss.load_to_weight
    print(f"\n  Improvement: {improvement:.1f}x better load/weight ratio!")


if __name__ == "__main__":
    main()
