import numpy as np
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

model = load_model_from_path(
    "/home/michael/University/Master/Thesis/adaptive-hexapod-control/mujoco-models/mk3/mk3.xml")
sim = MjSim(model)

viewer = MjViewer(sim)

sim_state = sim.get_state()

while True:
    sim.data.ctrl[:] = np.deg2rad(0)
    sim.data.ctrl[1] = np.deg2rad(50)
    sim.step()
    viewer.render()
