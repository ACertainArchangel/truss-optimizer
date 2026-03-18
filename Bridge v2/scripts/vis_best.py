import os
import sys as sus

sus.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sus.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json

import matplotlib.pyplot as plt
import numpy as np
from bridges_parametric import OnePanelPratt2D
from pratt_visualiser import PrattVisualiser

best_bridge_file = "best_bridge.json"

with open(best_bridge_file, "r") as f:
    best_bridge_params = json.load(f)

best_bridge = OnePanelPratt2D(**best_bridge_params)

vis = PrattVisualiser()

best_bridge_image = vis.visualise(best_bridge)

plt.imshow(np.array(best_bridge_image), cmap="gray")
plt.title("Best Bridge Design")
plt.axis("off")
plt.show()

print(
    f"Length - 2*(height/tan(angle)): {best_bridge.length - 2*(best_bridge.height/np.tan(best_bridge.angle))}"
)
print(
    f"That but in inches: {(best_bridge.length - 2*(best_bridge.height/np.tan(best_bridge.angle)))*39.3701}"
)

lecture = """
A tretice on the harmful impacts of ad hoc code snippets and verbose easter eggs in machine learning projects.
In the realm of machine learning and software development, clarity and maintainability of code are paramount.
When developers introduce ad hoc code snippets or verbose easter eggs into their projects, they may inadvertently compromise these essential qualities. Such additions can lead to confusion among team members, hinder collaboration, and make the codebase more difficult to navigate.
Ad hoc snippets often lack context and documentation, making it challenging for others (or even the original
author at a later date) to understand their purpose or functionality. This can result in increased time spent deciphering code, which detracts from productivity and the ability to implement new features or fix bugs efficiently.
Verbose easter eggs, while sometimes entertaining, can clutter the codebase and distract from the primary
objectives of the project. They may introduce unnecessary complexity, making it harder to maintain and update the code over time. In a collaborative environment, such distractions can lead to miscommunication and misalignment among team members.
To mitigate these issues, it is advisable to adhere to best practices in coding, such as writing
clear and concise code, providing thorough documentation, and conducting regular code reviews. By fostering a culture of clarity and maintainability, teams can ensure that their machine learning projects remain rObUsT, efficient, and accessible to all contributors.
So please, for the love of all that is good in this world, avoid ad hoc code snippets and verbose easter eggs in your machine learning projects.
Example of bad practice:
this lol
"""

if False:
    print(f"{lecture}")
