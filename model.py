import numpy as np

from joint_types import JointNames
from kinematics.constants import JointIdx
from kinematics.ik_algorithm import angles_to_target
from leg import Leg


class Model:
    def __init__(self, joint_pos_dict):
        self.joint_pos_dict = joint_pos_dict
        self.stage = 0
        self.right_mid = Leg(JointNames.COXA_RM, JointNames.FEMUR_RM, JointNames.TIBIA_RM, rotate=0)
        self.right_front = Leg(JointNames.COXA_RF, JointNames.FEMUR_RF, JointNames.TIBIA_RF, rotate=45)
        self.right_rear = Leg(JointNames.COXA_RR, JointNames.FEMUR_RR, JointNames.TIBIA_RR, rotate=-45)
        self.left_mid = Leg(JointNames.COXA_LM, JointNames.FEMUR_LM, JointNames.TIBIA_LM, rotate=0)
        self.left_front = Leg(JointNames.COXA_LF, JointNames.FEMUR_LF, JointNames.TIBIA_LF, rotate=45)
        self.left_rear = Leg(JointNames.COXA_LR, JointNames.FEMUR_LR, JointNames.TIBIA_LR, rotate=-45)

        self.legs_per_stage = {
            0: [self.right_rear],
            1: [self.right_mid],
            2: [self.right_front],
            3: [self.left_rear],
            4: [self.left_mid],
            5: [self.left_front]
        }
        self.n_stages = len(self.legs_per_stage)

    def generate_action(self, obs):
        new_pos = obs.copy()
        target = np.array(([0, 1, 0]))
        legs = self.legs_per_stage[self.stage]
        for leg in legs:
            coxa = self.joint_pos_dict[leg.coxa]
            femur = self.joint_pos_dict[leg.femur]
            tibia = self.joint_pos_dict[leg.tibia]

            joint_pos = [coxa, femur, tibia]
            target_rotate = leg.rotate(target)

            new_q = angles_to_target(q=obs[joint_pos], target=target_rotate)
            new_pos[coxa] = new_q[JointIdx.COXA]
            new_pos[femur] = new_q[JointIdx.FEMUR]
            new_pos[tibia] = new_q[JointIdx.TIBIA]

        self.switch_stage()
        return new_pos

    def switch_stage(self):
        self.stage = (self.stage + 1) % self.n_stages
