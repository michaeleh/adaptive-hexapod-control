import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

model = load_model_from_path(
    "/home/michael/University/Master/Thesis/adaptive-hexapod-control/mujoco-models/hexapod/hexapod.xml")
sim = MjSim(model)

viewer = MjViewer(sim)

sim_state = sim.get_state()
body_names = list(model.body_names)
i = 0
while True:
    i += 1
    sim.data.ctrl[:] = np.deg2rad(0)
    if i > 1000:
        sim.data.ctrl[1] = np.deg2rad(-40)

    sim.step()
    viewer.render()
