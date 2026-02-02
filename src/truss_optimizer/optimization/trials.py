"""
Run multiple optimization trials with different random seeds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from truss_optimizer.materials import Material
from truss_optimizer.optimization.optimizer import BridgeOptimizer
from truss_optimizer.optimization.result import OptimizationResult


@dataclass
class TrialResults:
    """
    Container for results from multiple optimization trials.
    
    Attributes:
        results: List of OptimizationResult from each trial
        material: Material used for all trials
        n_trials: Number of trials run
        
    Example:
        >>> results = run_optimization_trials(n_trials=100, material=material)
        >>> best = results.best()
        >>> print(f"Best load/weight: {best.load_to_weight:.1f}")
    """
    
    results: List[OptimizationResult] = field(default_factory=list)
    material: Optional[Material] = None
    n_trials: int = 0
    
    def best(self, by: str = 'load_to_weight') -> OptimizationResult:
        """
        Get the best result across all trials.
        
        Args:
            by: Metric to compare ('load_to_weight', 'critical_load', 'weight')
            
        Returns:
            Best OptimizationResult
        """
        if not self.results:
            raise ValueError("No results available")
        
        if by == 'load_to_weight':
            return max(self.results, key=lambda r: r.load_to_weight)
        elif by == 'critical_load':
            return max(self.results, key=lambda r: r.critical_load)
        elif by == 'weight':
            return min(self.results, key=lambda r: r.weight)
        else:
            raise ValueError(f"Unknown metric: {by}")
    
    def top_n(self, n: int = 10, by: str = 'load_to_weight') -> List[OptimizationResult]:
        """
        Get the top N results.
        
        Args:
            n: Number of results to return
            by: Metric to sort by
            
        Returns:
            List of top N results
        """
        if by == 'load_to_weight':
            sorted_results = sorted(self.results, key=lambda r: r.load_to_weight, reverse=True)
        elif by == 'critical_load':
            sorted_results = sorted(self.results, key=lambda r: r.critical_load, reverse=True)
        elif by == 'weight':
            sorted_results = sorted(self.results, key=lambda r: r.weight)
        else:
            raise ValueError(f"Unknown metric: {by}")
        
        return sorted_results[:n]
    
    def statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics across all trials.
        
        Returns:
            Dict with mean, std, min, max for key metrics
        """
        import numpy as np
        
        load_weights = [r.load_to_weight for r in self.results]
        loads = [r.critical_load for r in self.results]
        weights = [r.weight for r in self.results]
        
        def stats(values):
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        
        return {
            'load_to_weight': stats(load_weights),
            'critical_load': stats(loads),
            'weight': stats(weights),
        }
    
    def summary(self) -> str:
        """Generate summary of all trials."""
        stats = self.statistics()
        best = self.best()
        
        lines = [
            "=" * 60,
            f"  TRIAL RESULTS SUMMARY ({self.n_trials} trials)",
            "=" * 60,
            "",
            "  Load/Weight Ratio:",
            f"    Mean: {stats['load_to_weight']['mean']:.1f}",
            f"    Std:  {stats['load_to_weight']['std']:.1f}",
            f"    Best: {stats['load_to_weight']['max']:.1f}",
            "",
            "  Critical Load (N):",
            f"    Mean: {stats['critical_load']['mean']:.2f}",
            f"    Std:  {stats['critical_load']['std']:.2f}",
            f"    Best: {stats['critical_load']['max']:.2f}",
            "",
            "  Weight (N):",
            f"    Mean: {stats['weight']['mean']:.4f}",
            f"    Std:  {stats['weight']['std']:.4f}",
            f"    Min:  {stats['weight']['min']:.4f}",
            "",
            f"  Best Bridge:",
            f"    Load/Weight: {best.load_to_weight:.1f}",
            f"    Critical Load: {best.critical_load:.2f} N",
            f"    Weight: {best.weight:.4f} N",
            f"    Governing Mode: {best.governing_mode}",
            "",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save all trial results to a directory.
        
        Args:
            path: Directory path to save results
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        for i, result in enumerate(self.results):
            result.export(path / f"trial_{i:04d}.json")
        
        # Save summary
        summary = {
            'n_trials': self.n_trials,
            'statistics': self.statistics(),
            'best_index': self.results.index(self.best()),
        }
        
        with open(path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def run_optimization_trials(
    n_trials: int,
    material: Material,
    iterations: int = 5000,
    constraints: Optional[Dict] = None,
    save_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    **optimizer_kwargs,
) -> TrialResults:
    """
    Run multiple optimization trials with different random seeds.
    
    This is useful for exploring the design space and finding globally
    good solutions, since gradient descent can get stuck in local minima.
    
    Args:
        n_trials: Number of trials to run
        material: Material for all bridges
        iterations: Iterations per trial
        constraints: Parameter bounds (shared across trials)
        save_dir: Optional directory to save individual results
        verbose: Print progress
        **optimizer_kwargs: Additional arguments passed to BridgeOptimizer
        
    Returns:
        TrialResults containing all optimization results
        
    Example:
        >>> from truss_optimizer import run_optimization_trials, materials
        >>> 
        >>> results = run_optimization_trials(
        ...     n_trials=100,
        ...     material=materials.BalsaWood(),
        ...     iterations=5000,
        ...     save_dir='./trials'
        ... )
        >>> 
        >>> best = results.best()
        >>> print(f"Best load/weight: {best.load_to_weight:.1f}")
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    trial_results = TrialResults(material=material, n_trials=n_trials)
    
    if verbose:
        print(f"Running {n_trials} optimization trials...")
        print(f"Material: {material}")
        print("-" * 50)
    
    for i in range(n_trials):
        if verbose:
            print(f"Trial {i + 1}/{n_trials}...", end=" ", flush=True)
        
        # Create optimizer with random start
        optimizer = BridgeOptimizer.random_start(
            material=material,
            constraints=constraints,
            seed=i,
            **optimizer_kwargs,
        )
        
        # Run optimization
        result = optimizer.optimize(
            iterations=iterations,
            verbose=False,
        )
        
        trial_results.results.append(result)
        
        if verbose:
            print(f"Load/Weight = {result.load_to_weight:.1f}")
        
        # Save individual result
        if save_dir:
            result.export(save_dir / f"trial_{i:04d}.json")
    
    if verbose:
        print("-" * 50)
        best = trial_results.best()
        print(f"Best result: Load/Weight = {best.load_to_weight:.1f}")
        print(f"  Critical Load: {best.critical_load:.2f} N")
        print(f"  Weight: {best.weight:.4f} N")
    
    # Save summary
    if save_dir:
        trial_results.save(save_dir)
    
    return trial_results
