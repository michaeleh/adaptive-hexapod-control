import numpy as np
from abc import ABC, abstractmethod

from SNN.legs_heights_sensing import LegsHeightsModel
from utils import axis
from environment.hexapod_env import HexapodEnv
from environment.leg import all_legs


class AbstractLegHeightModel(ABC):
    def __init__(self, env: HexapodEnv):
        self.env = env

    @abstractmethod
    def get_legs_hs(self):
        pass


class SimLegHeightModel(AbstractLegHeightModel):
    def get_legs_hs(self):
        return np.array([self.env.get_pos(leg.coxa.value) * axis.z for leg in all_legs])


class NeuromorphicLegHeightModel(AbstractLegHeightModel):
    def __init__(self, env: HexapodEnv):
        super().__init__(env)
        self.sim_height_model = SimLegHeightModel(env)
        heights0 = self.sim_height_model.get_legs_hs().T[-1]
        self.model = LegsHeightsModel(heights0)
        self.previous = heights0
        self.history = []

    def get_legs_hs(self):
        return self.model.curr_val

    def update(self):
        curr_heights = self.sim_height_model.get_legs_hs().T[-1]
        height_change = curr_heights - self.previous
        # self.history.append(height_change)
        self.model.update(height_change)
        self.previous = curr_heights.copy()
