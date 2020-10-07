import numpy as np

from joint_types import JointNames

step_size = 20
dist = 10


class Leg:
    target_up = np.array([0, 0, -step_size])


class LegRM(Leg):
    coxa, femur, tibia = JointNames.COXA_RM, JointNames.FEMUR_RM, JointNames.TIBIA_RM
    target_forward = np.array([0, -step_size, step_size])


class LegLM(Leg):
    coxa, femur, tibia = JointNames.COXA_LM, JointNames.FEMUR_LM, JointNames.TIBIA_LM
    target_forward = np.array([0, -step_size, step_size])


class LegRF(Leg):
    coxa, femur, tibia = JointNames.COXA_RF, JointNames.FEMUR_RF, JointNames.TIBIA_RF
    target_forward = np.array([dist, -dist, step_size])


class LegLF(Leg):
    coxa, femur, tibia = JointNames.COXA_LF, JointNames.FEMUR_LF, JointNames.TIBIA_LF
    target_forward = np.array([dist, -dist, step_size])


class LegRR(Leg):
    coxa, femur, tibia = JointNames.COXA_RR, JointNames.FEMUR_RR, JointNames.TIBIA_RR
    target_forward = np.array([-dist, -dist, step_size])


class LegLR(Leg):
    coxa, femur, tibia = JointNames.COXA_LR, JointNames.FEMUR_LR, JointNames.TIBIA_LR
    target_forward = np.array([-dist, -dist, step_size])
