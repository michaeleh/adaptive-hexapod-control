from abc import ABC, abstractmethod
from enum import Enum
from itertools import cycle

from model.leg import LegRF, LegLF, LegRR, LegLM, LegLR, LegRM


class StageType(Enum):
    UP, ROTATE, DOWN, BACK = range(4)


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
            [LegRF(), LegRR(), LegLM()],
            [LegLF(), LegLR(), LegRM()]
        ])


class _1LegCycle(_Cycle):
    def get_legs_cycle(self):
        return cycle([
            [LegRF()],
            [LegRM()],
            [LegRR()],
            [LegLF()],
            [LegLM()],
            [LegLR()]
        ])


class _2LegCycle(_Cycle):
    def get_legs_cycle(self):
        return cycle([
            [LegRM(), LegLR()],
            [LegRF(), LegLM()],
            [LegLF(), LegRR()]
        ])
