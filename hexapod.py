from typing import Dict

from gym import Env
from mujoco_py import MjSim, MjViewer

from joint_types import JointNames
from leg import Leg


class Hexapod(Env):

    def __init__(self, model):
        self.model = model
        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)
        self.right_mid = Leg(JointNames.COXA_RM, JointNames.FEMUR_RM, JointNames.TIBIA_RM)
        self.right_front = Leg(JointNames.COXA_RF, JointNames.FEMUR_RF, JointNames.TIBIA_RF)
        self.right_rear = Leg(JointNames.COXA_RR, JointNames.FEMUR_RR, JointNames.TIBIA_RR)
        self.left_mid = Leg(JointNames.COXA_LM, JointNames.FEMUR_LM, JointNames.TIBIA_LM, left_side=True)
        self.left_front = Leg(JointNames.COXA_LF, JointNames.FEMUR_LF, JointNames.TIBIA_LF, left_side=True)
        self.left_rear = Leg(JointNames.COXA_LR, JointNames.FEMUR_LR, JointNames.TIBIA_LR, left_side=True)

        self.n_stages = 6
        self.stage = 0
        self.legs_per_stage = {
            0: [self.right_rear],
            1: [self.right_mid],
            2: [self.right_front],
            3: [self.left_rear],
            4: [self.left_mid],
            5: [self.left_front]
        }

    def step(self, action: Dict):
        legs = self.legs_per_stage[self.stage]
        for leg in legs:
            coxa = self.model.get_joint_qpos_addr(leg.coxa)
            femur = self.model.get_joint_qpos_addr(leg.femur)
            tibia = self.model.get_joint_qpos_addr(leg.tibia)
            self.sim.data.qpos[coxa] = leg.coxa_sign() * action['coxa']
            self.sim.data.qpos[femur] = leg.femur_sign() * action['femur']
            self.sim.data.qpos[tibia] = leg.tibia_sign() * action['tibia']

        if action['finished']:
            self.switch_stage()

        self.sim.forward()
        # self.sim.step()

    def reset(self):
        self.sim.reset()
        self.stage = 0

    def render(self, mode='human'):
        self.viewer.render()

    def switch_stage(self):
        self.stage = (self.stage + 1) % self.n_stages
