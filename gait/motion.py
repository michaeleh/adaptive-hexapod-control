import numpy as np
from abc import ABC, abstractmethod
from gait.motion_sync import MotionSync
from kinematics.ik_algorithm import angles_to_target
from gait.walking_cycles import _Cycle, StageType, _3LegCycle, _1LegCycle, _2LegCycle
from model.leg import forward_vec


class _Motion(ABC):
    def __init__(self, joint_pos_dict):
        self.joint_pos_dict = joint_pos_dict
        self.cycle = self.get_cycle()
        self.motion_sync = MotionSync(joint_pos_dict)

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
                # set new angles in relation to base position not in relation to current one
                new_pos[joint_pos], e = angles_to_target(q=np.zeros(3), target=leg.target_forward)

            if stage == StageType.DOWN:
                new_pos[joint_pos], e = angles_to_target(q=obs[joint_pos], target=-leg.target_up)

        if stage == StageType.SYNC:
            # sync body and all the legs toward direction of interest
            new_pos = self.motion_sync.sync_movement(
                qpos=obs,
                legs_to_move={leg: forward_vec for leg in legs}
            )
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
