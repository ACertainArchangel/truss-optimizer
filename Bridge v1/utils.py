def inches_to_meters(inches):
    return inches * 0.0254


def feet_to_meters(feet):
    return feet * 0.3048


import numpy as np
from scipy.optimize import root


def find_root_excluding(f, x_domain, exclude_range: tuple = None, max_attempts=50, tol=1e-8):
    """Find a root of f in x_domain, optionally excluding roots in exclude_range."""
    xmin, xmax = x_domain
    guesses = np.linspace(xmin, xmax, max_attempts)
    found_roots = []
    for x0 in guesses:
        sol = root(f, x0)
        if not sol.success:
            continue
        x = sol.x.item()
        if not np.isfinite(x):
            continue
        if any(abs(x - r) < tol for r in found_roots):
            continue
        if exclude_range is not None:
            a, b = exclude_range
            if a < x < b:
                continue
        return x
    raise RuntimeError("No root found outside the excluded range after trying all guesses.")
