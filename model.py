import mujoco_py as mjc

from joint_names import JointNames

import numpy as np
class Model:

    def __init__(self, xml_path):
        self.mjc_model = mjc.load_model_from_path(xml_path)

    def visualize(self):
        sim: mjc.MjSim = mjc.MjSim(self.mjc_model)
        viewer: mjc.MjViewer = mjc.MjViewer(sim)
        init_angles = {joint.value:np.pi/2 for joint in JointNames}
        for name, angle in init_angles.items():
            print(sim.data.qpos[self.qpos(name)])
            sim.data.qpos[self.qpos(name)] = sim.data.qpos[self.qpos(name)]+0.1
        sim.forward()  # Compute forward kinematics
        while True:
            # sim.step()
            viewer.render()

    def qpos(self, name):
        return self.mjc_model.get_joint_qpos_addr(name)
