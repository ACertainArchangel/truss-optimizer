import json
import math

from truss_optimizer import PrattTruss
from truss_optimizer.materials import OchromaWithEpoxy
from truss_optimizer.optimization.result import OptimizationResult
from truss_optimizer.trusses.pratt.truss import TrussParameters

trial_paths = [f"trials/final_pratt_bridge{i}.json" for i in range(100)]

trials = []
for path in trial_paths:
    with open(path, "r") as f:
        data = json.load(f)
    material = OchromaWithEpoxy()
    bridge = PrattTruss(params=TrussParameters.from_dict(data), material=material, K=0.5)
    trials.append(
        {
            "params": data,
            "critical_load": bridge.critical_load,
            "load_to_weight": bridge.load_to_weight,
            "governing_mode": bridge.governing_failure_mode,
        }
    )

best = max(trials, key=lambda t: t["load_to_weight"])
worst = min(trials, key=lambda t: t["load_to_weight"])
average_lw = sum(t["load_to_weight"] for t in trials) / len(trials)

best_index = trials.index(best)
print(f"  Best bridge found in trial {best_index}")

from truss_optimizer.optimization.result import OptimizationResult

best_result = OptimizationResult(
    params=best["params"],
    initial_params=best["params"],
    critical_load=best["critical_load"],
    weight=PrattTruss(
        params=TrussParameters.from_dict(best["params"]), material=OchromaWithEpoxy(), K=0.5
    ).weight,
    load_to_weight=best["load_to_weight"],
    governing_mode=best["governing_mode"],
    failure_modes={},
)
best_result.visualize()

print("=" * 60)
print("  TRIAL COMPARISON  (100 trials)")
print("=" * 60)
print(f"\n  Best  - Load/Weight: {best['load_to_weight']:>8.1f}")
print(f"          Critical Load: {best['critical_load']:>8.2f} N")
print(f"          Height:        {best['params'].get('height', 0)*1000:>8.1f} mm")
print(f"          Angle:         {math.degrees(best['params']['angle']):>8.1f} deg")
print(f"          Governing:     {best['governing_mode']}")
print(f"\n  Worst - Load/Weight: {worst['load_to_weight']:>8.1f}")
print(f"\n  Mean  - Load/Weight: {average_lw:>8.1f}")
print("=" * 60)
