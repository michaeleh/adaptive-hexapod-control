import numpy as np

from kinematics.ik_algorithm import angles_to_target
from kinematics.joint_kinematics import KinematicNumericImpl
from leg import LegRF, LegLR, LegRR, LegLM, LegRM, LegLF


class Model:
    def __init__(self, joint_pos_dict):
        self.joint_pos_dict = joint_pos_dict
        self.stage = 0

        self.legs_per_stage = [
            [LegRF()],
            [],
            [LegLF()],
            [],
            [LegRM()],
            [],
            [LegLM()],
            [],
            [LegRR()],
            [],
            [LegLR()],
            [],

        ]
        self.n_stages = len(self.legs_per_stage)
        self.swing_phase = 0
        self.k = KinematicNumericImpl()

    def generate_action(self, obs):
        new_pos = obs.copy()
        if self.stage == len(self.legs_per_stage):
            self.switch_stage()
            return np.zeros_like(obs)

        legs = self.legs_per_stage[self.stage]
        for leg in legs:
            coxa = self.joint_pos_dict[leg.coxa.value]
            femur = self.joint_pos_dict[leg.femur.value]
            tibia = self.joint_pos_dict[leg.tibia.value]

            joint_pos = [coxa, femur, tibia]
            if self.swing_phase == 0:
                target = leg.target_up
            if self.swing_phase == 1:
                target = leg.target_forward

            new_pos[joint_pos] = angles_to_target(q=obs[joint_pos], target=target, max_iter=10000,
                                                  error_thold=1e-10)

        self.switch_stage()
        return new_pos

    def switch_stage(self):
        self.swing_phase = (self.swing_phase + 1) % 2
        if self.swing_phase == 0:
            self.stage = (self.stage + 1) % self.n_stages
