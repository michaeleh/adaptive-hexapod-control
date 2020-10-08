from enum import Enum
from itertools import cycle

from kinematics.ik_algorithm import angles_to_target
from leg import LegRM, LegRF, LegLF, LegLR, LegRR, LegLM


class Model:
    def __init__(self, joint_pos_dict):
        self.joint_pos_dict = joint_pos_dict
        self.cycle = _StageCycle()

    def generate_action(self, obs):
        new_pos = obs.copy()
        legs, stage = self.cycle.get_next()

        for leg in legs:
            coxa = self.joint_pos_dict[leg.coxa.value]
            femur = self.joint_pos_dict[leg.femur.value]
            tibia = self.joint_pos_dict[leg.tibia.value]
            joint_pos = [coxa, femur, tibia]

            if stage == _StageType.UP:
                new_pos[joint_pos], e = angles_to_target(q=obs[joint_pos], target=leg.target_up)

            if stage == _StageType.ROTATE:
                new_pos[joint_pos], e = angles_to_target(q=obs[joint_pos], target=leg.target_forward)

            if stage == _StageType.DOWN:
                new_pos[joint_pos], e = angles_to_target(q=obs[joint_pos], target=-leg.target_up)
            if e > 0.1:
                new_pos[joint_pos] = 0

        return new_pos


class _StageType(Enum):
    UP, ROTATE, DOWN = range(3)


class _StageCycle:
    def __init__(self):
        self.stages_cycle = cycle([state for state in _StageType])
        self.legs_cycle = cycle([[LegRF(), LegRR(), LegLM()],
                                 [LegLF(), LegLR(), LegRM()]])
        self.legs = next(self.legs_cycle)
        self.stage = _StageType.UP

    def get_next(self):
        self.stage = next(self.stages_cycle)
        if self.stage == _StageType.UP:
            self.legs = next(self.legs_cycle)
        return self.legs, self.stage
