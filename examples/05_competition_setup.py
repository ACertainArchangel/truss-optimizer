"""
Example 5: Custom Material and Competition Setup

This example shows how to set up optimization for a specific
bridge building competition with custom material properties.
"""

import math
from truss_optimizer import BridgeOptimizer, PrattTruss
from truss_optimizer.materials import Custom
from truss_optimizer.utils import inches_to_meters


def main():
    print("=" * 70)
    print("  Competition Bridge Optimization")
    print("=" * 70)
    
    # Define custom material based on YOUR material testing
    # These values should come from actual material tests!
    print("\n  Step 1: Define Custom Material")
    print("  " + "-" * 50)
    
    # Example: You tested your balsa wood sheets and got:
    my_balsa = Custom(
        E=3.5e9,                    # Young's modulus from bending tests
        sigma_compression=10e6,     # From compression test
        sigma_tension=18e6,         # From tension test
        density=180.0,              # Measured density of your sheets
        name="My Tested Balsa"
    )
    
    print(f"  Material: {my_balsa.name}")
    print(f"  E = {my_balsa.E/1e9:.2f} GPa")
    print(f"  σ_compression = {my_balsa.sigma_compression/1e6:.1f} MPa")
    print(f"  σ_tension = {my_balsa.sigma_tension/1e6:.1f} MPa")
    print(f"  Density = {my_balsa.density:.0f} kg/m³")
    
    # Define competition constraints
    print("\n  Step 2: Competition Constraints")
    print("  " + "-" * 50)
    
    # Example competition rules:
    # - Span: exactly 18.5 inches
    # - Max height: 10 inches
    # - Min height: 4 inches
    # - Material thickness: 1/8" to 1/2"
    
    constraints = {
        # Fixed span
        'span': (inches_to_meters(18.5), inches_to_meters(18.5)),
        
        # Height limits from rules
        'height': (inches_to_meters(4.0), inches_to_meters(10.0)),
        
        # Angle limits (geometric reasoning)
        'angle': (math.radians(25), math.radians(70)),
        
        # Material stock constraints (what sheets you have)
        'incline_thickness': (inches_to_meters(1/8), inches_to_meters(3/8)),
        'diagonal_thickness': (inches_to_meters(1/8), inches_to_meters(1/4)),
        'mid_vert_thickness': (inches_to_meters(1/8), inches_to_meters(1/4)),
        'side_vert_thickness': (inches_to_meters(1/8), inches_to_meters(1/4)),
        'top_thickness': (inches_to_meters(1/8), inches_to_meters(3/8)),
        'bottom_thickness': (inches_to_meters(1/8), inches_to_meters(3/8)),
        
        # Depth constraints (width of your sheets)
        'incline_depth': (inches_to_meters(3/16), inches_to_meters(1/2)),
        'diagonal_depth': (inches_to_meters(3/16), inches_to_meters(3/8)),
        'mid_vert_depth': (inches_to_meters(3/16), inches_to_meters(3/8)),
        'side_vert_depth': (inches_to_meters(3/16), inches_to_meters(3/8)),
        'top_depth': (inches_to_meters(1/4), inches_to_meters(1/2)),
        'bottom_depth': (inches_to_meters(1/4), inches_to_meters(1/2)),
    }
    
    print("  Span: 18.5\" (fixed)")
    print("  Height: 4\" - 10\"")
    print("  Thickness: 1/8\" - 3/8\"")
    print("  Depth: 3/16\" - 1/2\"")
    
    # Account for fixed weight (epoxy, connectors)
    # Estimate based on previous builds
    fixed_weight = 0.5  # Newtons (about 50 grams)
    
    print(f"\n  Fixed weight (epoxy, etc.): {fixed_weight:.2f} N")
    
    # Create optimizer
    print("\n  Step 3: Optimization")
    print("  " + "-" * 50)
    
    # Start with reasonable initial values
    bridge = PrattTruss(
        span=inches_to_meters(18.5),
        height=inches_to_meters(6.0),
        angle=math.radians(35),
        material=my_balsa
    )
    
    optimizer = BridgeOptimizer(
        bridge=bridge,
        objective='load_to_weight',
        constraints=constraints,
        learning_rate=0.001,
        fixed_cost=fixed_weight,  # Account for epoxy weight
    )
    
    # Run optimization
    result = optimizer.optimize(
        iterations=5000,
        verbose=True,
        log_interval=1000
    )
    
    # Results
    print("\n  Step 4: Results")
    print("  " + "=" * 50)
    
    p = result.params
    
    print(f"\n  OPTIMIZED DESIGN:")
    print(f"  Height:       {p.get('height', 0)/0.0254:.3f}\"")
    print(f"  Angle:        {math.degrees(p.get('angle', 0)):.1f}°")
    
    print(f"\n  Member Dimensions (thickness × depth in inches):")
    members = ['incline', 'diagonal', 'mid_vert', 'side_vert', 'top', 'bottom']
    for m in members:
        t = p.get(f'{m}_thickness', 0) / 0.0254
        d = p.get(f'{m}_depth', 0) / 0.0254
        print(f"    {m:12}: {t:.3f}\" × {d:.3f}\"")
    
    print(f"\n  PERFORMANCE:")
    print(f"    Critical Load:  {result.critical_load:.1f} N ({result.critical_load*0.2248:.1f} lbf)")
    print(f"    Total Weight:   {result.weight:.3f} N ({result.weight*0.2248:.3f} lbf)")
    print(f"    Load/Weight:    {result.load_to_weight:.1f}")
    print(f"    Governing Mode: {result.governing_mode}")
    
    # Save for building
    result.export("competition_bridge.json")
    print(f"\n  Design saved to: competition_bridge.json")
    
    # Practical recommendations
    print("\n  Step 5: Building Recommendations")
    print("  " + "-" * 50)
    print("  • Round dimensions to available stock sizes")
    print("  • Add 10-20% safety factor to member sizes")
    print("  • Reinforce joints with gusset plates")
    print("  • Test material samples before building")
    print("  • Weigh components during construction")


if __name__ == "__main__":
    main()
