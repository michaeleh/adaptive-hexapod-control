import numpy as np
from typing import Dict

from kinematics.ik_algorithm import angles_to_target
from model.leg import Leg, all_legs


class MotionSync:
    def __init__(self, joint_pos_dict: Dict):
        """

        :param joint_pos_dict: mapping between joint and qpos
        """
        self.joint_pos_dict = joint_pos_dict
        self.body_xyz = [0, 1, 2]

    def sync_movement(self, qpos, legs_to_move: Dict[Leg, np.array]):
        """
        move legs and sync body movement and other legs using inverse kinematics
        :param qpos: current qpos of the model
        :param legs_to_move: dict from leg 2 move to destination in NED coordinates where foot-tip is the center.
        :return: new qpos synchronized with all aspects
        """
        new_qpos = qpos.copy()
        n_legs_to_stay = len(all_legs) - len(legs_to_move)
        legs_delta_pos = [np.zeros(3)] * n_legs_to_stay + list(legs_to_move.values())
        mean_walking_dir = np.array(legs_delta_pos).mean(axis=0)
        # move body in the average direction
        new_qpos[self.body_xyz] += mean_walking_dir
        # adjust other legs to make foot tip still
        for leg in all_legs:
            coxa = self.joint_pos_dict[leg.coxa.value]
            femur = self.joint_pos_dict[leg.femur.value]
            tibia = self.joint_pos_dict[leg.tibia.value]
            joint_pos = [coxa, femur, tibia]
            new_qpos[joint_pos], e = angles_to_target(q=qpos[joint_pos], target=-leg.rotate(mean_walking_dir))
        return new_qpos
