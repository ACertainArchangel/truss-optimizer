connector_cost = 0.165

epoxy_cost = 0.515

fixed_cost = connector_cost + epoxy_cost

CONTINUE_FROM_EXISTING = True

import json

from bridges_parametric import OnePanelPratt2D
from materials import (
    ConservativeBalsaWood,
    HighDensityBalsaWood,
    LowDensityBalsaWood,
    OchromaWithEpoxy,
    OchromaWood,
)
from pratt_optimizer import make_and_train_pratt
from pratt_visualiser import PrattVisualiser

_MATERIALS_MAIN = [
    ConservativeBalsaWood,
    HighDensityBalsaWood,
    LowDensityBalsaWood,
    OchromaWood,
    OchromaWithEpoxy,
]

_MATERIAL_MAIN = 4


def fully_abstracted_pratt_workflow(
    seed: int = 42, verbose: bool = True, terminate_turtle: bool = True, flat_factor: float = 0
):
    """Run full Pratt bridge optimization, visualization, and export."""

    material = _MATERIALS_MAIN[_MATERIAL_MAIN]()

    initial_params, final_params = make_and_train_pratt(
        seed=seed,
        iterations=7000,
        lr=0.001,
        verbose=verbose,
        material=material,
        flat_factor=flat_factor,
        fixed_cost=epoxy_cost,
    )

    density = material.density

    for param in ["material", "density"]:

        if param in initial_params:
            del initial_params[param]

        if param in final_params:
            del final_params[param]

    vis = PrattVisualiser()

    bridge_image_init = vis.visualise(OnePanelPratt2D(**initial_params))
    bridge_image_final = vis.visualise(OnePanelPratt2D(**final_params))

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(np.array(bridge_image_init), cmap="gray")
    axs[0].set_title("Initial Bridge Design")
    axs[0].axis("off")
    axs[1].imshow(np.array(bridge_image_final), cmap="gray")
    axs[1].set_title("Final Bridge Design")
    axs[1].axis("off")

    plt.savefig(f"trials/pratt_bridge_comparison{seed}.png")

    if verbose:
        print(f"Saved bridge comparison image as 'trials/pratt_bridge_comparison{seed}.png'")

    if True:
        print("\n")

        final_bridge = OnePanelPratt2D(**final_params)

        with open(f"trials/final_pratt_bridge{seed}.json", "w") as f:
            f.write(final_params.__repr__().replace("'", '"'))

        if verbose:
            print(f"Saved final bridge design to 'trials/final_pratt_bridge{seed}.json'")

        def dict_pretty_print(d: dict):
            """Pretty print a dict as JSON, sorted ascending with nulls last."""

            import json
            from collections import OrderedDict

            def sort_key(item):
                key, value = item

                if value is None:
                    return (1, key)
                return (0, value)

            ordered = OrderedDict(sorted(d.items(), key=sort_key))

            print(json.dumps(ordered, indent=4))

        final_bridge = OnePanelPratt2D(**final_params)

        dict_to_be_printed = final_bridge.get_failure_mode_dict()

        if density is not None:
            material_weight = final_bridge.get_total_volume() * density * 9.81
            dict_to_be_printed["weight_N"] = material_weight + fixed_cost

            if fixed_cost != 0.0:
                dict_to_be_printed["material_weight_N"] = material_weight
                dict_to_be_printed["fixed_cost_per_plane_N"] = fixed_cost

        if verbose:
            dict_pretty_print(dict_to_be_printed)

        if terminate_turtle:
            vis.shutdown()


if __name__ == "__main__":
    iters = 2000

    if CONTINUE_FROM_EXISTING:
        import os

        existing_files = [
            f
            for f in os.listdir("trials")
            if f.startswith("final_pratt_bridge") and f.endswith(".json")
        ]

        num_existing = len(existing_files)

    else:
        num_existing = 0

    if num_existing >= iters:
        print(
            f"Found {num_existing} existing trials, which is >= the requested {iters} trials. No new trials to run."
        )
        exit(0)

    import tqdm

    seeds = [i for i in range(num_existing, iters)]

    for seed in tqdm.tqdm(seeds):

        fully_abstracted_pratt_workflow(
            seed=seed, verbose=False, terminate_turtle=True if seed == seeds[-1] else False
        )

    import post_process_parameters_perfectly

    post_process_parameters_perfectly.main("in")

    final_params = json.load(open("final_plans.json"))["means"]

    from pratt_analyse_v2 import make_and_report

    material = _MATERIALS_MAIN[_MATERIAL_MAIN]()

    make_and_report(
        OnePanelPratt2D(**final_params, material=material),
        density=material.density,
        special_message="BRIDGE FAILURE ANALYSIS FOR AVERAGED PARAMETERS ",
    )
