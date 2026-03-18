import json

from bridges_parametric import OnePanelPratt2D
from materials import ConservativeBalsaWood, HighDensityBalsaWood, LowDensityBalsaWood, OchromaWood
from pratt_optimizer import make_and_train_pratt
from pratt_visualiser import PrattVisualiser

_MATERIALS_MAIN = [ConservativeBalsaWood, HighDensityBalsaWood, LowDensityBalsaWood, OchromaWood]
_MATERIAL_MAIN = 3


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
    print(f"Saved bridge comparison image as 'trials/pratt_bridge_comparison{seed}.png'")
    if True:
        print("\n")
        final_bridge = OnePanelPratt2D(**final_params)
        with open(f"trials/final_pratt_bridge{seed}.json", "w") as f:
            f.write(final_params.__repr__().replace("'", '"'))
        print(f"Saved final bridge design to 'trials/final_pratt_bridge{seed}.json'")

        def dict_pretty_print(d: dict):
            """Pretty print a dict as sorted JSON, nulls last."""
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
            dict_to_be_printed["weight_N"] = final_bridge.get_total_volume() * density * 9.81
        if verbose:
            dict_pretty_print(dict_to_be_printed)
        if terminate_turtle:
            vis.shutdown()


if __name__ == "__main__":
    import tqdm  # مِكَرٍّ مِفَرٍّ مُقْبِلٍ مُدْبِرٍ مَعًا كَجُلْمُودِ صَخْرٍ حَطَّهُ السَّيْلُ مِنْ عَلِ

    seeds = [i for i in range(2)]
    for seed in tqdm.tqdm(seeds):
        fully_abstracted_pratt_workflow(
            seed=seed, verbose=False, terminate_turtle=True if seed == seeds[-1] else False
        )
    import post_process_parameters_perfectly

    post_process_parameters_perfectly.main("in")
    final_params = json.load(open("final_plans.json"))["means"]
    from pratt_analyse_v2 import make_and_report

    material = _MATERIALS_MAIN[_MATERIAL_MAIN]()
    make_and_report(OnePanelPratt2D(**final_params, material=material), density=material.density)
