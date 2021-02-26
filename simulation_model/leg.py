import numpy as np

from simulation_model.joint_types import JointNames

step_size = 80
up_vec = np.array([0, 0, 10])  # up vector
forward_vec = np.array([0, step_size, 0])  # direction vector


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
        :return: [side (Left,Right), Location (Fron, Mid, Rear)]
        """
        pos = self.coxa.value.split('_')[1]
        return list(pos)


class LegRM(Leg):
    coxa, femur, tibia = JointNames.COXA_RM, JointNames.FEMUR_RM, JointNames.TIBIA_RM


class LegLM(Leg):
    coxa, femur, tibia = JointNames.COXA_LM, JointNames.FEMUR_LM, JointNames.TIBIA_LM


class LegRF(Leg):
    coxa, femur, tibia = JointNames.COXA_RF, JointNames.FEMUR_RF, JointNames.TIBIA_RF
    angle = -45
    target_forward = rotate_vec(forward_vec, angle)


class LegLF(Leg):
    coxa, femur, tibia = JointNames.COXA_LF, JointNames.FEMUR_LF, JointNames.TIBIA_LF
    angle = -45
    target_forward = rotate_vec(forward_vec, angle)


class LegRR(Leg):
    coxa, femur, tibia = JointNames.COXA_RR, JointNames.FEMUR_RR, JointNames.TIBIA_RR
    angle = 45
    target_forward = rotate_vec(forward_vec, angle)


class LegLR(Leg):
    coxa, femur, tibia = JointNames.COXA_LR, JointNames.FEMUR_LR, JointNames.TIBIA_LR
    angle = 45
    target_forward = rotate_vec(forward_vec, angle)


leg_rf = LegRF()
leg_rm = LegRM()
leg_rr = LegRR()
leg_lf = LegLF()
leg_lm = LegLM()
leg_lr = LegLR()
all_legs = [leg_rf, leg_rm, leg_rr, leg_lf, leg_lm, leg_lr]
