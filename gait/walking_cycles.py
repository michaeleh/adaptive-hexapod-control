from abc import ABC, abstractmethod
from enum import Enum
from itertools import cycle

from model.leg import leg_rf, leg_rm, leg_rr, leg_lf, leg_lm, leg_lr


class StageType(Enum):
    UP, ROTATE, DOWN, SYNC = range(4)


ROTATION_ANGLE = 2


class _Cycle(ABC):
    def __init__(self):
        self.stages_cycle = cycle([state for state in StageType])
        self.legs_cycle = self.get_legs_cycle()
        self.legs = next(self.legs_cycle)
        self.stage = StageType.UP

    def get_next(self):
        self.stage = next(self.stages_cycle)
        if self.stage == StageType.UP:
            self.legs = next(self.legs_cycle)
        return self.legs, self.stage

    @abstractmethod
    def get_legs_cycle(self):
        pass


class _3LegCycle(_Cycle):

    def get_legs_cycle(self):
        return cycle([
            [leg_rf, leg_rr, leg_lm],
            [leg_lf, leg_lr, leg_rm]
        ])


class _1LegCycle(_Cycle):
    def get_legs_cycle(self):
        return cycle([
            [leg_rf],
            [leg_rm],
            [leg_rr],
            [leg_lf],
            [leg_lm],
            [leg_lr]
        ])


class _2LegCycle(_Cycle):
    def get_legs_cycle(self):
        return cycle([
            [leg_rm, leg_lr],
            [leg_rf, leg_lm],
            [leg_lf, leg_rr]
        ])