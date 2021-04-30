from enum import Enum

import numpy as np

from environment.joint_types import JointNames
from utils.coordinates import polar2xy
from utils.vectors import rotate_vec


class _DirectionManager:
    def __init__(self):
        self.up_vec = np.array([0, 0, -0.05])  # up vector we want to go up from the ee thus
        self.r = 0.04
        self.theta = np.deg2rad(180)  # constant direction
        self.theta_change = 0


direction_manager = _DirectionManager()


class Leg:
    """
    Legs creation for joint names, and rotated direction vector
    """
    target_up = direction_manager.up_vec.copy()
    coxa, femur, tibia = '', '', ''
    angle = 0

    def rotate(self, vec):
        return rotate_vec(vec, self.angle)

    def position(self):
        """
        :return: [side (Left,Right), Location (Front, Mid, Rear)]
        """
        pos = self.coxa.value.split('_')[1]
        return list(pos)

    @property
    def target_forward(self):
        x, y = polar2xy(direction_manager.r, direction_manager.theta + direction_manager.theta_change + self.angle)
        return np.array([x, y, 0])


class LegRM(Leg):
    coxa, femur, tibia, ee = JointNames.COXA_RM, JointNames.FEMUR_RM, JointNames.TIBIA_RM, ''  # EENames.EE_RM


class LegLM(Leg):
    angle = np.deg2rad(180)
    coxa, femur, tibia, ee = JointNames.COXA_LM, JointNames.FEMUR_LM, JointNames.TIBIA_LM, ''  # EENames.EE_LM


class LegRF(Leg):
    coxa, femur, tibia, ee = JointNames.COXA_RF, JointNames.FEMUR_RF, JointNames.TIBIA_RF, ''  # EENames.EE_RF
    angle = np.deg2rad(-45)


class LegLF(Leg):
    coxa, femur, tibia, ee = JointNames.COXA_LF, JointNames.FEMUR_LF, JointNames.TIBIA_LF, ''  # , EENames.EE_LF
    angle = np.deg2rad(-135)


class LegRR(Leg):
    coxa, femur, tibia, ee = JointNames.COXA_RR, JointNames.FEMUR_RR, JointNames.TIBIA_RR, ''  # EENames.EE_RR
    angle = np.deg2rad(45)


class LegLR(Leg):
    coxa, femur, tibia, ee = JointNames.COXA_LR, JointNames.FEMUR_LR, JointNames.TIBIA_LR, ''  # EENames.EE_LR
    angle = np.deg2rad(135)


leg_rf = LegRF()
leg_rm = LegRM()
leg_rr = LegRR()
leg_lf = LegLF()
leg_lm = LegLM()
leg_lr = LegLR()
all_legs = [leg_rf, leg_rm, leg_rr, leg_lf, leg_lm, leg_lr]

side_to_leg_dict = {
    'RF': leg_rf,
    'RM': leg_rm,
    'RR': leg_rr,
    'LF': leg_lf,
    'LM': leg_lm,
    'LR': leg_lr
}
