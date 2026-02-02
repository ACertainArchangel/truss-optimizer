"""
Example 3: Comparing Different Materials

This example shows how to compare bridge performance
across different materials.
"""

import math
from truss_optimizer import BridgeOptimizer, PrattTruss, materials
from truss_optimizer.utils import inches_to_meters


def optimize_for_material(material, iterations=2000):
    """Optimize a bridge for a specific material."""
    bridge = PrattTruss(
        span=inches_to_meters(18.5),
        height=inches_to_meters(6.0),
        angle=math.radians(30),
        material=material
    )
    
    optimizer = BridgeOptimizer(
        bridge=bridge,
        objective='load_to_weight',
        learning_rate=0.001,
        constraints={
            'span': (inches_to_meters(18.5), inches_to_meters(18.5)),
            'height': (inches_to_meters(4.0), inches_to_meters(10.0)),
            'angle': (math.radians(25), math.radians(65)),
        }
    )
    
    result = optimizer.optimize(iterations=iterations, verbose=False)
    return result


def main():
    print("=" * 70)
    print("  Material Comparison Study")
    print("=" * 70)
    
    # Materials to compare
    test_materials = [
        ("Balsa Wood", materials.BalsaWood()),
        ("Light Balsa", materials.LightBalsaWood()),
        ("Heavy Balsa", materials.HeavyBalsaWood()),
        ("PLA (3D Print)", materials.PLA()),
        ("Carbon PLA", materials.CarbonFiberPLA()),
        ("Aluminum", materials.Aluminum()),
    ]
    
    print("\n  Material Properties:")
    print("  " + "-" * 65)
    print(f"  {'Material':<15} {'E (GPa)':>10} {'σ_c (MPa)':>12} {'σ_t (MPa)':>12} {'ρ (kg/m³)':>12}")
    print("  " + "-" * 65)
    
    for name, mat in test_materials:
        print(f"  {name:<15} {mat.E/1e9:>10.1f} {mat.sigma_compression/1e6:>12.1f} "
              f"{mat.sigma_tension/1e6:>12.1f} {mat.density:>12.0f}")
    
    print("\n  Optimizing each material (this may take a minute)...")
    print("  " + "-" * 65)
    
    results = []
    for name, material in test_materials:
        print(f"  Optimizing {name}...", end=" ", flush=True)
        result = optimize_for_material(material)
        results.append((name, material, result))
        print(f"Load/Weight = {result.load_to_weight:.1f}")
    
    # Summary table
    print("\n  RESULTS SUMMARY:")
    print("  " + "=" * 65)
    print(f"  {'Material':<15} {'Load (N)':>10} {'Weight (N)':>12} {'L/W Ratio':>12} {'Governing Mode':<20}")
    print("  " + "-" * 65)
    
    # Sort by load-to-weight ratio
    results.sort(key=lambda x: x[2].load_to_weight, reverse=True)
    
    for name, mat, result in results:
        print(f"  {name:<15} {result.critical_load:>10.1f} {result.weight:>12.4f} "
              f"{result.load_to_weight:>12.1f} {result.governing_mode:<20}")
    
    print("  " + "=" * 65)
    
    # Winner
    winner_name, _, winner_result = results[0]
    print(f"\n  🏆 Best Material: {winner_name}")
    print(f"     Load/Weight Ratio: {winner_result.load_to_weight:.1f}")
    
    # Key insight
    print("\n  📊 Key Insight:")
    print("  " + "-" * 50)
    
    # Compare balsa vs PLA
    balsa_lw = next(r[2].load_to_weight for r in results if r[0] == "Balsa Wood")
    pla_lw = next(r[2].load_to_weight for r in results if r[0] == "PLA (3D Print)")
    
    print(f"  Balsa wood achieves {balsa_lw/pla_lw:.1f}x better load/weight")
    print("  ratio than PLA due to its exceptional strength-to-weight.")
    print("\n  For model bridge competitions, low-density materials")
    print("  like balsa wood are hard to beat!")


if __name__ == "__main__":
    main()
