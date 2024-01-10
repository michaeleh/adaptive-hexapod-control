import numpy as np
from abc import ABC, abstractmethod

from SNN.legs_heights_sensing import LegsHeightsModel
from kinematics.joint_kinematics import KinematicNumericImpl
from utils import axis
from environment.hexapod_env import HexapodEnv
from environment.leg import all_legs

fk = KinematicNumericImpl()


class AbstractLegHeightModel(ABC):
    def __init__(self, env: HexapodEnv):
        self.env = env
        self.heights0 = None
        self.d_from_default = None
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
    def __init__(self, n_neurons, tau, transform, env: HexapodEnv):
        super().__init__(env)
        self.sim_height_model = SimLegHeightModel(env)
        self.heights0 = self.sim_height_model.get_legs_hs().T[-1]
        self.d_from_default = (self.heights0 - fk.calc_xyz(np.zeros(3))[-1]).mean()
        self.model = LegsHeightsModel(n_neurons, tau, transform, self.heights0)
        self.previous = self.heights0
        self.history = []

    def get_legs_hs(self):
        hs = self.model.curr_val
        return hs

    def update(self):
        curr_heights = self.sim_height_model.get_legs_hs().T[-1]
        height_change = curr_heights - self.previous
        self.history.append(height_change.copy())
        self.model.update(height_change)

        self.previous = curr_heights.copy()

