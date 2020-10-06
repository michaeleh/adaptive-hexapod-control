import numpy as np

from joint_types import JointNames


class Leg:
    def __init__(self, coxa: JointNames, femur: JointNames, tibia: JointNames, rotate: int):
        self.theta = np.deg2rad(rotate)
        self.coxa = coxa.value
        self.femur = femur.value
        self.tibia = tibia.value

    def rotate(self, target: np.array):
        x, y, z = target
        theta = self.theta
        return np.array([
            x * np.cos(theta) - y * np.sin(theta),
            x * np.sin(theta) + y * np.cos(theta),
            z
        ])
