import numpy as np

from joint_types import JointNames

step_size = 40
forward_vec = np.array([0, step_size, 0])


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
    target_up = np.array([0, 0, step_size])
    target_forward = forward_vec.copy()


class LegRM(Leg):
    coxa, femur, tibia = JointNames.COXA_RM, JointNames.FEMUR_RM, JointNames.TIBIA_RM


class LegLM(Leg):
    coxa, femur, tibia = JointNames.COXA_LM, JointNames.FEMUR_LM, JointNames.TIBIA_LM


class LegRF(Leg):
    coxa, femur, tibia = JointNames.COXA_RF, JointNames.FEMUR_RF, JointNames.TIBIA_RF
    target_forward = rotate_vec(forward_vec, -45)


class LegLF(Leg):
    coxa, femur, tibia = JointNames.COXA_LF, JointNames.FEMUR_LF, JointNames.TIBIA_LF
    target_forward = rotate_vec(forward_vec, -45)


class LegRR(Leg):
    coxa, femur, tibia = JointNames.COXA_RR, JointNames.FEMUR_RR, JointNames.TIBIA_RR
    target_forward = rotate_vec(forward_vec, 45)


class LegLR(Leg):
    coxa, femur, tibia = JointNames.COXA_LR, JointNames.FEMUR_LR, JointNames.TIBIA_LR
    target_forward = rotate_vec(forward_vec, 45)
