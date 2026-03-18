import json

from src.truss_optimizer.materials import (  # TODO: REMOVE src. WHEN WE ARE DONE TESTING
    BalsaOG,
    BalsaWood,
    Epoxy,
    rule_of_mixtures,
)
from src.truss_optimizer.trusses import PrattTrussV1  # TODO: REMOVE src. WHEN WE ARE DONE TESTING
from src.truss_optimizer.utils import (  # TODO: REMOVE src. WHEN WE ARE DONE TESTING
    fig_to_image,
    inches_to_meters,
    meters_to_inches,
)

path = "competition_bridge.json"
loaded_bridge = json.load(open(path, "r"))
params = loaded_bridge["parameters"]
params.pop("span", None)

objectivebridge = PrattTrussV1(
    material=rule_of_mixtures(
        BalsaWood(), Epoxy(), 0.5, "Balsa Laminate", ideality_coefficient=0.90
    ),  # 50% balsa, 50% epoxy by volume. Generally, laminates lose 5-15% effeciency so taking the average and setting eta=0.9 is a good rule of thumb.
    length=inches_to_meters(18.5),
    parameters=params,
    k=0.5,
    fixed_cost=2.4,
)

ogbridge = PrattTrussV1(
    material=BalsaOG(),
    length=inches_to_meters(18.5),
    parameters=params,
    k=0.5,  # Was 1 in the original analysis but tension was the problem so it didn't matter.
    fixed_cost=0.183,
)

print(objectivebridge.critical_load())
print(objectivebridge.weight())
print(objectivebridge.load_to_weight_ratio())

print(ogbridge.critical_load())
print(ogbridge.weight())
print(ogbridge.load_to_weight_ratio())
