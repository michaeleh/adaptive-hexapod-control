from enum import Enum

import numpy as np

from environment.joint_types import JointNames

up_vec = np.array([0, 0, 50])  # up vector
forward_vec = np.array([0, 80, 0])  # direction vector


class Side(Enum):
    R, L = range(2)


def rotate_vec(vec, deg):
    rad = np.deg2rad(deg)
    '''
    x' = x cos θ − y sin θ
    y' = x sin θ + y cos θ
    '''
    x = vec[0]
    y = vec[1]
    z = vec[2]
    return np.array([
        x * np.cos(rad) - y * np.sin(rad),
        x * np.sin(rad) + y * np.cos(rad),
        z
    ])


class Leg:
    """
    Legs creation for joint names, and rotated direction vector
    """
    target_up = up_vec.copy()
    target_forward = forward_vec.copy()
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
    def side(self):
        pos = self.position()
        if pos[0] == 'L':
            return Side.L
        else:
            return Side.R


class LegRM(Leg):
    coxa, femur, tibia, ee = JointNames.COXA_RM, JointNames.FEMUR_RM, JointNames.TIBIA_RM, ''  # EENames.EE_RM


class LegLM(Leg):
    angle = 180
    coxa, femur, tibia, ee = JointNames.COXA_LM, JointNames.FEMUR_LM, JointNames.TIBIA_LM, ''  # EENames.EE_LM
    target_forward = rotate_vec(forward_vec, angle)


class LegRF(Leg):
    coxa, femur, tibia, ee = JointNames.COXA_RF, JointNames.FEMUR_RF, JointNames.TIBIA_RF, ''  # EENames.EE_RF
    angle = -45
    target_forward = rotate_vec(forward_vec, angle)


class LegLF(Leg):
    coxa, femur, tibia, ee = JointNames.COXA_LF, JointNames.FEMUR_LF, JointNames.TIBIA_LF, ''  # , EENames.EE_LF
    angle = -135
    target_forward = rotate_vec(forward_vec, angle)


class LegRR(Leg):
    coxa, femur, tibia, ee = JointNames.COXA_RR, JointNames.FEMUR_RR, JointNames.TIBIA_RR, ''  # EENames.EE_RR
    angle = 45
    target_forward = rotate_vec(forward_vec, angle)


class LegLR(Leg):
    coxa, femur, tibia, ee = JointNames.COXA_LR, JointNames.FEMUR_LR, JointNames.TIBIA_LR, ''  # EENames.EE_LR
    angle = 135
    target_forward = rotate_vec(forward_vec, angle)


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
