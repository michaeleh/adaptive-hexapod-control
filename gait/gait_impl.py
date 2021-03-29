from abc import ABC, abstractmethod

import numpy as np

from gait.walking_cycles import _Cycle, StageType, _3LegCycle, _1LegCycle, _2LegCycle
from kinematics.ik_algorithm import angles_to_target


class _Motion(ABC):
    def __init__(self, joint_pos_dict):
        self.joint_pos_dict = joint_pos_dict
        self.cycle = self.get_cycle()

    @abstractmethod
    def get_cycle(self) -> _Cycle:
        pass

    def generate_action(self, obs):
        """
        Generate action according to current cycle and leg group
        """
        new_pos = {}
        legs, stage = self.cycle.get_next()

        for leg in legs:
            # get qpos of each joint
            coxa = self.joint_pos_dict[leg.coxa.value]
            femur = self.joint_pos_dict[leg.femur.value]
            tibia = self.joint_pos_dict[leg.tibia.value]
            joint_pos = [coxa, femur, tibia]
            if stage == StageType.UP:
                q, _ = angles_to_target(q=obs[joint_pos], target=leg.target_up)

            if stage == StageType.FORWARD:
                # set new angles in relation to base position not in relation to current one
                q, _ = angles_to_target(q=np.zeros(3), target=leg.target_forward + leg.target_up)

            if stage == StageType.DOWN:
                q, _ = angles_to_target(q=obs[joint_pos], target=-leg.target_up)
            if stage == StageType.RETURN:
                q = np.zeros(3)

            new_pos[leg.coxa.value], new_pos[leg.femur.value], new_pos[leg.tibia.value] = q

        return new_pos


class TripodMotion(_Motion):
    def get_cycle(self) -> _Cycle:
        return _3LegCycle()


class WaveMotion(_Motion):
    def get_cycle(self) -> _Cycle:
        return _1LegCycle()


class RippleMotion(_Motion):
    def get_cycle(self) -> _Cycle:
        return _2LegCycle()
