"""
Example 1: Basic Bridge Optimization

This example demonstrates the simplest way to use Truss Optimizer
to design an efficient balsa wood bridge.
"""

import math
from truss_optimizer import BridgeOptimizer, PrattTruss, materials
from truss_optimizer.utils import inches_to_meters


def main():
    print("=" * 60)
    print("  Truss Optimizer - Basic Example")
    print("=" * 60)
    
    # Step 1: Define the material
    # Using standard balsa wood - great for model bridges!
    material = materials.BalsaWood()
    print(f"\nMaterial: {material}")
    print(f"  Young's Modulus: {material.E/1e9:.2f} GPa")
    print(f"  Compressive Strength: {material.sigma_compression/1e6:.1f} MPa")
    print(f"  Tensile Strength: {material.sigma_tension/1e6:.1f} MPa")
    print(f"  Density: {material.density:.0f} kg/m³")
    
    # Step 2: Create initial bridge design
    # Competition standard: 18.5" span
    bridge = PrattTruss(
        span=inches_to_meters(18.5),   # ~470 mm
        height=inches_to_meters(6.0),   # Starting height
        angle=math.radians(30),         # 30 degree inclines
        material=material
    )
    
    print(f"\nInitial Bridge:")
    print(f"  Span: {bridge.span*1000:.1f} mm")
    print(f"  Height: {bridge.height*1000:.1f} mm")
    print(f"  Angle: {math.degrees(bridge.angle):.1f}°")
    print(f"  Initial Load/Weight: {bridge.load_to_weight:.1f}")
    
    # Step 3: Create the optimizer
    # We want to maximize the load-to-weight ratio
    optimizer = BridgeOptimizer(
        bridge=bridge,
        objective='load_to_weight',  # Maximize load/weight
        learning_rate=0.001,
        constraints={
            # Fix span (competition requirement)
            'span': (inches_to_meters(18.5), inches_to_meters(18.5)),
            # Allow height to vary
            'height': (inches_to_meters(4.0), inches_to_meters(10.0)),
            # Allow angle to vary  
            'angle': (math.radians(25), math.radians(65)),
        }
    )
    
    # Step 4: Run the optimization
    print("\n" + "-" * 60)
    print("Starting optimization...")
    print("-" * 60)
    
    result = optimizer.optimize(
        iterations=3000,
        verbose=True,
        log_interval=500
    )
    
    # Step 5: View results
    print("\n" + "=" * 60)
    print("  OPTIMIZATION RESULTS")
    print("=" * 60)
    
    print(f"\n  Critical Load:     {result.critical_load:>10.2f} N")
    print(f"  Weight:            {result.weight:>10.4f} N")
    print(f"  Load/Weight Ratio: {result.load_to_weight:>10.1f}")
    print(f"\n  Governing Failure: {result.governing_mode}")
    
    # Show optimized geometry
    p = result.params
    print(f"\n  Optimized Geometry:")
    print(f"    Height: {p.get('height', 0)*1000:.1f} mm ({p.get('height', 0)/0.0254:.2f} in)")
    print(f"    Angle:  {math.degrees(p.get('angle', 0)):.1f}°")
    
    # Step 6: Save the result
    result.export("basic_bridge.json")
    print(f"\n  Saved to: basic_bridge.json")
    
    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
