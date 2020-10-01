from joint_types import JointNames


class Leg:
    def __init__(self, coxa: JointNames, femur: JointNames, tibia: JointNames, left_side=False):
        self.left_side = left_side
        self.coxa = coxa.value
        self.femur = femur.value
        self.tibia = tibia.value

    def coxa_sign(self):
        return -1 if self.left_side else 1

    def tibia_sign(self):
        return -1

    def femur_sign(self):
        return -1
