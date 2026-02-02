"""
Advanced example: Running 100 parallel optimization trials to find the global optimum.

This demonstrates:
- Running multiple trials with different random starting points
- Parallel execution using multiprocessing for speed
- Specifying custom initial parameters for some trials
- Taking the best result from all trials
- Visualization of the best design found
- Complete overkill and possibly time wasting

WARNING: This script is almost guaranteed to make your computer parts go like...
CPU/GPUs: Please stop you're killing me.
Fans: wawawawawaaaaaaahrrrrrrrrr
Ram: Bro I'm stuffed
Heat sinks: 🔥I'm melting🔥

And in most cases is overkill since the loss landscape isn't that complex and you will only 
squeese out like 60 more competition points but hey at least
it shows how to do it if you really want to find the absolute best design possible and 
you have the hardware to back it up so there you go I like run on sentances. *gasps*
"""

import json
import math
import multiprocessing as mp

from truss_optimizer import BridgeOptimizer, PrattTruss, materials
from truss_optimizer.optimization.trials import TrialResults
from truss_optimizer.utils.units import inches_to_meters


def run_single_trial(seed: int, material, constraints, iterations, fixed_cost, initial_params=None):
    """
    Run a single optimization trial. This function is picklable for multiprocessing.
    
    Args:
        seed: Random seed for this trial
        material: Material to use
        constraints: Parameter bounds
        iterations: Number of optimization steps
        fixed_cost: Fixed weight for fasteners etc.
        initial_params: Optional dict of specific starting parameters
    
    Returns:
        OptimizationResult from this trial
    """
    if initial_params is not None:
        # Use provided initial parameters
        bridge = PrattTruss(**initial_params, material=material)
        optimizer = BridgeOptimizer(
            bridge=bridge,
            objective='load_to_weight',
            constraints=constraints,
            fixed_cost=fixed_cost,
        )
    else:
        # Random start with given seed
        optimizer = BridgeOptimizer.random_start(
            material=material,
            constraints=constraints,
            seed=seed,
            objective='load_to_weight',
            fixed_cost=fixed_cost,
        )
    
    result = optimizer.optimize(iterations=iterations, verbose=False)
    return result


def run_parallel_trials(
    n_trials: int,
    material,
    constraints: dict,
    iterations: int = 10000,
    fixed_cost: float = 0.2,
    n_workers: int = None,
    custom_starts: list = None,
) -> TrialResults:
    """
    Run multiple optimization trials in parallel using multiprocessing.
    
    Args:
        n_trials: Total number of trials to run
        material: Material for all bridges
        constraints: Parameter bounds
        iterations: Steps per trial
        fixed_cost: Fixed weight (kg)
        n_workers: Number of parallel workers (defaults to CPU count)
        custom_starts: Optional list of dicts with specific initial params for some trials
    
    Returns:
        TrialResults containing all optimization results
    """
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    print(f"Running {n_trials} trials in parallel with {n_workers} workers...")
    print(f"Material: {material}")
    print(f"Iterations per trial: {iterations}")
    print("-" * 60)
    
    # Build task list: each task is a complete tuple of all arguments
    tasks = []
    n_custom = len(custom_starts) if custom_starts else 0
    
    # Add custom starts first
    if custom_starts:
        for i, params in enumerate(custom_starts):
            tasks.append((i, material, constraints, iterations, fixed_cost, params))
    
    # Fill remaining with random starts
    for seed in range(n_custom, n_trials):
        tasks.append((seed, material, constraints, iterations, fixed_cost, None))
    
    # Run in parallel
    results = []
    with mp.Pool(processes=n_workers) as pool:
        # Use starmap to unpack all arguments
        results = pool.starmap(run_single_trial, tasks)
    
    # Wrap in TrialResults
    trial_results = TrialResults(results=results, material=material, n_trials=n_trials)
    
    print("-" * 60)
    print(trial_results.summary())
    
    return trial_results


if __name__ == "__main__":
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Competition constraints
    thickness_bounds = (inches_to_meters(0.1), inches_to_meters(0.5))  # 2.54mm - 12.7mm max
    depth_bounds = (inches_to_meters(0.1), inches_to_meters(0.5))
    
    constraints = {
        # Thickness constraints for each member type
        'incline_thickness': thickness_bounds,
        'diagonal_thickness': thickness_bounds,
        'mid_vert_thickness': thickness_bounds,
        'side_vert_thickness': thickness_bounds,
        'top_thickness': thickness_bounds,
        'bottom_thickness': thickness_bounds,
        # Depth constraints for each member type
        'incline_depth': depth_bounds,
        'diagonal_depth': depth_bounds,
        'mid_vert_depth': depth_bounds,
        'side_vert_depth': depth_bounds,
        'top_depth': depth_bounds,
        'bottom_depth': depth_bounds,
        # Geometric constraints
        'angle': (math.radians(25), math.radians(65)),
        'height': (inches_to_meters(4), inches_to_meters(6)),
    }
    
    # =========================================================================
    # CUSTOM STARTING POINTS
    # These are specific configurations we want to try in addition to random starts.
    # This is useful if you have intuition about good designs or want to refine
    # a previously found solution.
    # =========================================================================
    
    custom_starts = [
        # A "tall and steep" configuration
        {
            'span': 0.47,
            'height': inches_to_meters(6),      # Max height
            'angle': math.radians(55),          # Steep inclines
            'incline_thickness': inches_to_meters(0.5),
            'incline_depth': inches_to_meters(0.5),
            'diagonal_thickness': inches_to_meters(0.25),
            'diagonal_depth': inches_to_meters(0.25),
            'mid_vert_thickness': inches_to_meters(0.5),
            'mid_vert_depth': inches_to_meters(0.5),
            'side_vert_thickness': inches_to_meters(0.25),
            'side_vert_depth': inches_to_meters(0.25),
            'top_thickness': inches_to_meters(0.5),
            'top_depth': inches_to_meters(0.5),
            'bottom_thickness': inches_to_meters(0.5),
            'bottom_depth': inches_to_meters(0.5),
        },
        # A "short and shallow" configuration
        {
            'span': 0.47,
            'height': inches_to_meters(4),      # Min height
            'angle': math.radians(35),          # Shallow inclines
            'incline_thickness': inches_to_meters(0.4),
            'incline_depth': inches_to_meters(0.4),
            'diagonal_thickness': inches_to_meters(0.2),
            'diagonal_depth': inches_to_meters(0.3),
            'mid_vert_thickness': inches_to_meters(0.4),
            'mid_vert_depth': inches_to_meters(0.4),
            'side_vert_thickness': inches_to_meters(0.2),
            'side_vert_depth': inches_to_meters(0.2),
            'top_thickness': inches_to_meters(0.4),
            'top_depth': inches_to_meters(0.4),
            'bottom_thickness': inches_to_meters(0.4),
            'bottom_depth': inches_to_meters(0.3),
        },
        # A "balanced" configuration
        {
            'span': 0.47,
            'height': inches_to_meters(5),
            'angle': math.radians(45),
            'incline_thickness': inches_to_meters(0.5),
            'incline_depth': inches_to_meters(0.5),
            'diagonal_thickness': inches_to_meters(0.15),
            'diagonal_depth': inches_to_meters(0.2),
            'mid_vert_thickness': inches_to_meters(0.5),
            'mid_vert_depth': inches_to_meters(0.5),
            'side_vert_thickness': inches_to_meters(0.15),
            'side_vert_depth': inches_to_meters(0.15),
            'top_thickness': inches_to_meters(0.5),
            'top_depth': inches_to_meters(0.5),
            'bottom_thickness': inches_to_meters(0.4),
            'bottom_depth': inches_to_meters(0.35),
        },
    ]
    
    # =========================================================================
    # RUN PARALLEL OPTIMIZATION
    # =========================================================================
    
    trial_results = run_parallel_trials(
        n_trials=100,                           # 100 total trials
        material=materials.BalsaWood(),
        constraints=constraints,
        iterations=10000,                       # 10,000 steps per trial
        fixed_cost=0.2,                         # 0.2 kg for fasteners/glue
        n_workers=None,                         # Use all CPU cores
        custom_starts=custom_starts,            # Include our custom starting points
    )
    
    # =========================================================================
    # ANALYZE RESULTS
    # =========================================================================
    
    # Get the best result
    best = trial_results.best()
    
    print("\n" + "=" * 60)
    print("  BEST DESIGN FOUND")
    print("=" * 60)
    print(f"\nLoad/Weight Ratio: {best.load_to_weight:.1f}")
    print(f"Critical Load:     {best.critical_load:.2f} N")
    print(f"Weight:            {best.weight:.4f} kg")
    print(f"Governing Mode:    {best.governing_mode}")
    print("\nOptimized Parameters:")
    print(json.dumps(best.params, indent=4))
    
    # Show top 5 results
    print("\n" + "-" * 60)
    print("Top 5 Results:")
    for i, result in enumerate(trial_results.top_n(5)):
        print(f"  {i+1}. Load/Weight = {result.load_to_weight:.1f}, "
              f"Load = {result.critical_load:.1f} N, "
              f"Weight = {result.weight:.4f} kg")
    
    # Visualize the best design
    print("\nDisplaying visualization of best design...")
    best.visualize(show=True)
