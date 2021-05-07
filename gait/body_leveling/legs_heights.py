import numpy as np
from abc import ABC, abstractmethod

from SNN.legs_heights_sensing import LegsHeightsModel
from utils import axis
from environment.hexapod_env import HexapodEnv
from environment.leg import all_legs


class AbstractLegHeightModel(ABC):
    def __init__(self, env: HexapodEnv):
        self.env = env
        self.heights0 = None
        self.torso0 = env.get_pos('torso')

    @abstractmethod
    def get_legs_hs(self):
        pass

    def torso_h_change(self):
        return self.env.get_pos('torso') - self.torso0


class SimLegHeightModel(AbstractLegHeightModel):
    def __init__(self, env: HexapodEnv):
        super().__init__(env)
        self.heights0 = self.get_legs_hs().T[-1]

    def get_legs_hs(self):
        return np.array([self.env.get_pos(leg.coxa.value) * axis.z for leg in all_legs])


class NeuromorphicLegHeightModel(AbstractLegHeightModel):
    def __init__(self, env: HexapodEnv):
        super().__init__(env)
        self.sim_height_model = SimLegHeightModel(env)
        self.heights0 = self.sim_height_model.get_legs_hs().T[-1]

        self.model = LegsHeightsModel(self.heights0)
        self.previous = self.heights0
        self.history = []
        self.targets = []

    def get_legs_hs(self):
        hs = self.model.curr_val
        return hs

    def update(self):
        curr_heights = self.sim_height_model.get_legs_hs().T[-1]
        height_change = curr_heights - self.previous
        self.history.append(height_change.copy())
        self.model.update(height_change)

        self.previous = curr_heights.copy()

        d_from_ground = (self.model.curr_val - self.heights0).min()
        goal = self.heights0.mean() * 1.3 + d_from_ground
        self.targets.append(goal)
