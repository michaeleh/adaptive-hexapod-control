from simulation_model.sim_templates.abstract_sim_template import AbstractSimTemplate


class SimCollision(AbstractSimTemplate):
    def __init__(self):
        super().__init__()
        self.contacts = []

    def eval(self):
        pass
