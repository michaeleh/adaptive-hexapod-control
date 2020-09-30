import mujoco_py as mjc
import numpy as np


class Model:

    def __init__(self, xml_path):
        self.mjc_model = mjc.load_model_from_path(xml_path)

    def visualize(self):
        sim: mjc.MjSim = mjc.MjSim(self.mjc_model)
        viewer: mjc.MjViewer = mjc.MjViewer(sim)

        while True:
            actions = np.random.uniform(low=-0.001, high=0.001, size=sim.data.ctrl.shape)[:]

            sim.data.ctrl[:] = actions[:]
            sim.step()
            viewer.render()
