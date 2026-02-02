"""
Example 2: Multiple Trials for Global Optimization

Running multiple trials with different random seeds helps find
better global solutions, since gradient descent can get stuck
in local minima.
"""

from truss_optimizer import run_optimization_trials, materials
from truss_optimizer.utils import inches_to_meters
import math


def main():
    print("=" * 60)
    print("  Multi-Trial Optimization")
    print("=" * 60)
    
    # Define material
    material = materials.BalsaWood()
    
    # Define constraints (competition rules)
    constraints = {
        'span': (inches_to_meters(18.5), inches_to_meters(18.5)),  # Fixed
        'height': (inches_to_meters(4.0), inches_to_meters(10.0)),
        'angle': (math.radians(25), math.radians(65)),
    }
    
    # Run multiple trials
    print(f"\nRunning optimization trials...")
    print(f"Material: {material}")
    print(f"Constraints: span fixed at 18.5\", height 4-10\", angle 25-65°")
    print("-" * 60)
    
    results = run_optimization_trials(
        n_trials=20,           # Run 20 different random starts
        material=material,
        iterations=2000,       # 2000 iterations per trial
        constraints=constraints,
        learning_rate=0.001,
        verbose=True,          # Show progress
    )
    
    # Print statistics
    print("\n" + results.summary())
    
    # Get the best result
    best = results.best()
    
    print("\n  BEST BRIDGE DETAILS:")
    print("-" * 40)
    p = best.params
    print(f"  Height:      {p.get('height', 0)*1000:.1f} mm")
    print(f"  Angle:       {math.degrees(p.get('angle', 0)):.1f}°")
    print(f"  Top Depth:   {p.get('top_depth', 0)*1000:.2f} mm")
    print(f"  Bot Depth:   {p.get('bottom_depth', 0)*1000:.2f} mm")
    
    # Show top 5 results
    print("\n  TOP 5 BRIDGES:")
    print("-" * 40)
    for i, r in enumerate(results.top_n(5), 1):
        print(f"  {i}. Load/Weight = {r.load_to_weight:.1f}, "
              f"Critical = {r.critical_load:.1f} N, "
              f"Mode = {r.governing_mode}")
    
    # Save best result
    best.export("best_trial_bridge.json")
    print(f"\n  Best bridge saved to: best_trial_bridge.json")


if __name__ == "__main__":
    main()
