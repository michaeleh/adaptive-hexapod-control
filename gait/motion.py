from abc import ABC, abstractmethod

from kinematics.ik_algorithm import angles_to_target
from gait.walking_cycles import _Cycle, StageType, _3LegCycle, _1LegCycle, _2LegCycle


class _Motion(ABC):
    def __init__(self, joint_pos_dict):
        self.joint_pos_dict = joint_pos_dict
        self.cycle = self.get_cycle()

    @abstractmethod
    def get_cycle(self) -> _Cycle:
        pass

    def generate_action(self, obs):
        new_pos = obs.copy()
        legs, stage = self.cycle.get_next()

        for leg in legs:
            coxa = self.joint_pos_dict[leg.coxa.value]
            femur = self.joint_pos_dict[leg.femur.value]
            tibia = self.joint_pos_dict[leg.tibia.value]
            joint_pos = [coxa, femur, tibia]

            if stage == StageType.UP:
                new_pos[joint_pos], e = angles_to_target(q=obs[joint_pos], target=leg.target_up)

            if stage == StageType.ROTATE:
                new_pos[joint_pos], e = angles_to_target(q=obs[joint_pos], target=leg.target_forward)

            if stage == StageType.DOWN:
                new_pos[joint_pos], e = angles_to_target(q=obs[joint_pos], target=-leg.target_up)

            if stage == StageType.BACK:
                new_pos[joint_pos] = 0
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
