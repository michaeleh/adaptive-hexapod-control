import mujoco_py as mjc


class Model:

    def __init__(self, xml_path):
        self.mjc_model = mjc.load_model_from_path(xml_path)

    def visualize(self):
        simulation: mjc.MjSim = mjc.MjSim(self.mjc_model)
        viewer: mjc.MjViewer = mjc.MjViewer(simulation)
        while True:
            simulation.step()
            viewer.render()
