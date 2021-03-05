import numpy as np
from typing import Dict

from kinematics.ik_algorithm import angles_to_target
from simulation_model.leg import all_legs


class MotionSync:
    """
    Syncing the body position to the leg placement and movement
    """

    def __init__(self, joint_pos_dict: Dict):
        """

        :param joint_pos_dict: mapping between joint and qpos
        """
        self.joint_pos_dict = joint_pos_dict
        self.body_xyz = [0, 1, 2]  # joint position in qpos

    def sync_movement(self, qpos, axis_change):
        """
        move legs and sync body movement and other legs using inverse kinematics
        :param qpos: current qpos of the simulation_model
        :param legs_to_move: dict from leg 2 move to destination in NED coordinates where foot-tip is the center.
        :return: new qpos synchronized with all aspects
        """

        # move body in the average direction
        new_qpos = qpos.copy()
        mean_dir = axis_change / len(all_legs)
        # forward motion on the ground
        new_qpos[self.body_xyz] += mean_dir
        # adjust other legs to make foot tip still
        for leg in all_legs:
            coxa = self.joint_pos_dict[leg.coxa.value]
            femur = self.joint_pos_dict[leg.femur.value]
            tibia = self.joint_pos_dict[leg.tibia.value]
            joint_pos = [coxa, femur, tibia]
            new_qpos[joint_pos], e = angles_to_target(q=qpos[joint_pos], target=-leg.rotate(mean_dir))

        return new_qpos
