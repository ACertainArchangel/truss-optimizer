"""
Main optimizer class for truss bridge design.

This module provides the BridgeOptimizer class which uses PyTorch automatic
differentiation for gradient-based optimization of truss parameters.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from truss_optimizer.core.truss import PrattTruss, TrussParameters
from truss_optimizer.materials import Material
from truss_optimizer.optimization.torch_physics import (
    DTYPE,
    compute_max_load,
    compute_volume,
    compute_weight,
)
from truss_optimizer.optimization.result import OptimizationResult
from truss_optimizer.utils.units import inches_to_meters


class BridgeOptimizer:
    """
    Gradient-based optimizer for truss bridge design.
    
    Uses PyTorch automatic differentiation to efficiently optimize bridge
    parameters by maximizing load capacity, minimizing weight, or optimizing
    the load-to-weight ratio.
    
    Args:
        bridge: Initial PrattTruss to optimize (or use initial_params)
        initial_params: Dict of initial parameter values
        material: Material properties
        objective: Optimization objective:
            - 'max_load': Maximize critical failure load
            - 'min_weight': Minimize total weight
            - 'load_to_weight': Maximize load-to-weight ratio (default)
        constraints: Dict of parameter bounds {name: (min, max)}
        learning_rate: Adam optimizer learning rate
        K: Effective length factor for buckling
        fixed_cost: Additional fixed weight (N), e.g., for connectors
        device: Torch device ('cpu' or 'cuda')
        
    Example:
        >>> from truss_optimizer import BridgeOptimizer, PrattTruss, materials
        >>> 
        >>> bridge = PrattTruss(span=0.47, height=0.15, material=materials.BalsaWood())
        >>> optimizer = BridgeOptimizer(bridge, objective='load_to_weight')
        >>> result = optimizer.optimize(iterations=5000)
        >>> 
        >>> print(f"Load/Weight: {result.load_to_weight:.1f}")
        >>> result.visualize()
    """
    
    # Default parameter bounds (can be overridden)
    DEFAULT_BOUNDS = {
        'angle': (math.radians(20), math.radians(70)),
        'height': (inches_to_meters(3), inches_to_meters(12)),
        'thickness': (inches_to_meters(0.1), inches_to_meters(0.5)),
        'depth': (inches_to_meters(0.1), inches_to_meters(0.5)),
    }
    
    # Parameters that should be fixed by default
    DEFAULT_FIXED = {'E', 'sigma_compression', 'sigma_tension', 'density', 'span', 'length'}
    
    def __init__(
        self,
        bridge: Optional[PrattTruss] = None,
        initial_params: Optional[Dict[str, float]] = None,
        material: Optional[Material] = None,
        objective: str = 'load_to_weight',
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        learning_rate: float = 0.001,
        K: float = 1.0,
        fixed_cost: float = 0.0,
        device: Optional[str] = None,
    ):
        self.device = device or 'cpu'
        self.learning_rate = learning_rate
        self.K = K
        self.fixed_cost = fixed_cost
        
        # Validate objective
        valid_objectives = {'max_load', 'min_weight', 'load_to_weight'}
        if objective not in valid_objectives:
            raise ValueError(f"objective must be one of {valid_objectives}")
        self.objective = objective
        
        # Get initial parameters
        if bridge is not None:
            self.initial_params = bridge.to_dict()
            self.material = bridge.material or material
        elif initial_params is not None:
            self.initial_params = dict(initial_params)
            self.material = material
        else:
            raise ValueError("Must provide either bridge or initial_params")
        
        # Add material properties to params
        if self.material:
            self.initial_params['E'] = self.material.E
            self.initial_params['sigma_compression'] = self.material.sigma_compression
            self.initial_params['sigma_tension'] = self.material.sigma_tension
            self.density = self.material.density
        else:
            self.density = self.initial_params.get('density', 7850.0)
        
        # Setup constraints/bounds
        self.constraints = self._setup_constraints(constraints)
        
        # Setup trainable parameters
        self._setup_trainable_params()
        
        # Create optimizer
        trainable_tensors = [
            v for k, v in self.trainable_params.items()
            if k not in self.DEFAULT_FIXED and v.requires_grad
        ]
        self.torch_optimizer = torch.optim.Adam(trainable_tensors, lr=learning_rate)
        
        # Training history
        self.loss_history: List[float] = []
        self.load_history: List[float] = []
        self.param_history: Dict[str, List[float]] = {
            name: [] for name in self.trainable_params
        }
        
        # Best parameters tracking
        self.best_loss = float('inf')
        self.best_params: Dict[str, float] = {}
        self.best_iteration = 0
    
    def _setup_constraints(
        self, 
        constraints: Optional[Dict[str, Tuple[float, float]]]
    ) -> Dict[str, Tuple[float, float]]:
        """Setup parameter constraints/bounds."""
        bounds: Dict[str, Tuple[float, float]] = {}
        
        # Apply defaults
        thickness_params = [
            'incline_thickness', 'diagonal_thickness', 'mid_vert_thickness',
            'side_vert_thickness', 'top_thickness', 'bottom_thickness'
        ]
        depth_params = [
            'incline_depth', 'diagonal_depth', 'mid_vert_depth',
            'side_vert_depth', 'top_depth', 'bottom_depth'
        ]
        
        for param in thickness_params:
            bounds[param] = self.DEFAULT_BOUNDS['thickness']
        for param in depth_params:
            bounds[param] = self.DEFAULT_BOUNDS['depth']
        
        bounds['angle'] = self.DEFAULT_BOUNDS['angle']
        bounds['height'] = self.DEFAULT_BOUNDS['height']
        
        # Fix span/length by default
        span = self.initial_params.get('span', self.initial_params.get('length', 0.47))
        bounds['span'] = (span, span)
        bounds['length'] = (span, span)
        
        # Override with user constraints
        if constraints:
            bounds.update(constraints)
        
        return bounds
    
    def _setup_trainable_params(self) -> None:
        """Create trainable PyTorch tensors for all parameters."""
        self.trainable_params: Dict[str, torch.Tensor] = {}
        
        for name, value in self.initial_params.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Determine if parameter is trainable
            is_fixed = name in self.DEFAULT_FIXED
            
            # Check if bounds make it fixed (min == max)
            if name in self.constraints:
                min_val, max_val = self.constraints[name]
                if min_val == max_val:
                    is_fixed = True
            
            tensor = torch.tensor(
                value,
                dtype=DTYPE,
                device=self.device,
                requires_grad=not is_fixed
            )
            self.trainable_params[name] = tensor
    
    def _get_params_dict(self) -> Dict[str, torch.Tensor]:
        """Get parameters dict for physics functions."""
        # Handle span/length naming
        params = dict(self.trainable_params)
        if 'span' in params and 'length' not in params:
            params['length'] = params['span']
        return params
    
    def _compute_loss(self) -> torch.Tensor:
        """Compute the loss function based on objective."""
        params = self._get_params_dict()
        
        max_load, _ = compute_max_load(
            angle=params['angle'],
            height=params['height'],
            length=params['length'],
            incline_thickness=params['incline_thickness'],
            diagonal_thickness=params['diagonal_thickness'],
            mid_vert_thickness=params['mid_vert_thickness'],
            side_vert_thickness=params['side_vert_thickness'],
            top_thickness=params['top_thickness'],
            bottom_thickness=params['bottom_thickness'],
            incline_depth=params['incline_depth'],
            diagonal_depth=params['diagonal_depth'],
            mid_vert_depth=params['mid_vert_depth'],
            side_vert_depth=params['side_vert_depth'],
            top_depth=params['top_depth'],
            bottom_depth=params['bottom_depth'],
            E=params['E'],
            sigma_compression=params['sigma_compression'],
            sigma_tension=params['sigma_tension'],
            K=self.K,
        )
        
        if self.objective == 'max_load':
            return -max_load  # Minimize negative load = maximize load
        
        # Compute weight for other objectives
        weight = compute_weight(
            density=self.density,
            fixed_cost=self.fixed_cost,
            **params,
        )
        
        # Clamp to avoid division issues
        max_load = torch.clamp(max_load, min=1e-6)
        weight = torch.clamp(weight, min=1e-12)
        
        if self.objective == 'min_weight':
            return weight
        else:  # load_to_weight
            return -max_load / weight  # Minimize negative ratio = maximize ratio
    
    def _apply_bounds(self) -> None:
        """Clamp parameters to their bounds."""
        for name, (min_val, max_val) in self.constraints.items():
            if name in self.trainable_params:
                with torch.no_grad():
                    self.trainable_params[name].clamp_(min_val, max_val)
    
    def _train_step(self, iteration: int) -> float:
        """Execute single training step."""
        self.torch_optimizer.zero_grad()
        loss = self._compute_loss()
        loss_val = loss.item()
        
        # Track best
        if loss_val < self.best_loss:
            self.best_loss = loss_val
            self.best_iteration = iteration
            self.best_params = {
                name: tensor.detach().item()
                for name, tensor in self.trainable_params.items()
            }
        
        # Backward pass
        loss.backward()
        self.torch_optimizer.step()
        self._apply_bounds()
        
        # Record history
        self.loss_history.append(loss_val)
        for name, tensor in self.trainable_params.items():
            self.param_history[name].append(tensor.item())
        
        return loss_val
    
    def optimize(
        self,
        iterations: int = 5000,
        verbose: bool = True,
        log_interval: int = 500,
        callback: Optional[Callable[[int, float, Dict], None]] = None,
    ) -> OptimizationResult:
        """
        Run the optimization.
        
        Args:
            iterations: Number of optimization iterations
            verbose: Print progress updates
            log_interval: Iterations between progress prints
            callback: Optional function called each iteration with (iter, loss, params)
            
        Returns:
            OptimizationResult with optimized parameters and analysis
            
        Example:
            >>> result = optimizer.optimize(iterations=5000, verbose=True)
            >>> print(result.summary())
        """
        if verbose:
            print(f"Starting optimization with objective: {self.objective}")
            print(f"Material: {self.material}")
            print("-" * 50)
        
        for iteration in range(iterations):
            loss = self._train_step(iteration)
            
            if callback:
                current_params = {
                    name: tensor.item() 
                    for name, tensor in self.trainable_params.items()
                }
                callback(iteration, loss, current_params)
            
            if verbose and (iteration + 1) % log_interval == 0:
                if self.objective == 'load_to_weight':
                    print(f"Iteration {iteration + 1:5d}: Load/Weight = {-loss:.2f}")
                elif self.objective == 'max_load':
                    print(f"Iteration {iteration + 1:5d}: Load = {-loss:.2f} N")
                else:
                    print(f"Iteration {iteration + 1:5d}: Weight = {loss:.4f} N")
        
        if verbose:
            print("-" * 50)
            print(f"Optimization complete. Best at iteration {self.best_iteration + 1}")
        
        return self._create_result(iterations)
    
    def _create_result(self, iterations: int) -> OptimizationResult:
        """Create OptimizationResult from current state."""
        # Use best parameters
        params = self.best_params
        
        # Handle span/length
        if 'span' not in params and 'length' in params:
            params['span'] = params['length']
        
        # Create bridge with best params
        bridge = PrattTruss.from_dict(params, material=self.material)
        
        # Get failure modes
        failure_modes = bridge.get_failure_modes()
        
        return OptimizationResult(
            params=params,
            initial_params=self.initial_params,
            critical_load=bridge.critical_load,
            weight=bridge.weight + self.fixed_cost,
            load_to_weight=bridge.critical_load / (bridge.weight + self.fixed_cost),
            governing_mode=bridge.governing_failure_mode,
            failure_modes=failure_modes,
            history={
                'loss': self.loss_history,
                **self.param_history,
            },
            iterations=iterations,
            final_loss=self.best_loss,
            best_iteration=self.best_iteration,
        )
    
    @classmethod
    def random_start(
        cls,
        material: Material,
        constraints: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "BridgeOptimizer":
        """
        Create optimizer with randomized initial parameters.
        
        Args:
            material: Material for the bridge
            constraints: Parameter bounds
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to BridgeOptimizer
            
        Returns:
            BridgeOptimizer with random initial parameters
            
        Example:
            >>> optimizer = BridgeOptimizer.random_start(
            ...     material=materials.BalsaWood(),
            ...     seed=42
            ... )
            >>> result = optimizer.optimize()
        """
        if seed is not None:
            random.seed(seed)
        
        # Setup bounds
        bounds = cls.DEFAULT_BOUNDS.copy()
        if constraints:
            bounds.update(constraints)
        
        # Generate random parameters
        params = TrussParameters()
        param_dict = asdict(params)
        
        for name in param_dict:
            if name in bounds:
                min_val, max_val = bounds[name]
                param_dict[name] = random.uniform(min_val, max_val)
            elif 'thickness' in name:
                min_val, max_val = bounds['thickness']
                param_dict[name] = random.uniform(min_val, max_val)
            elif 'depth' in name:
                min_val, max_val = bounds['depth']
                param_dict[name] = random.uniform(min_val, max_val)
        
        return cls(
            initial_params=param_dict,
            material=material,
            constraints=constraints,
            **kwargs,
        )
