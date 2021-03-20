from abc import ABC, abstractmethod
from enum import Enum
from itertools import cycle

from environment.leg import leg_rf, leg_rm, leg_rr, leg_lf, leg_lm, leg_lr


class StageType(Enum):
    UP, FORWARD, DOWN, RETURN, WAIT = range(5)


class _Cycle(ABC):
    """
    Defining leg groups for gaits cycle.
    """

    def __init__(self):
        self.stages_cycle = cycle([state for state in StageType] + [StageType.WAIT]*1)  # leg swing cycle
        self.legs_cycle = self.get_legs_cycle()
        self.legs = next(self.legs_cycle)
        self.stage = StageType.UP

    def get_next(self):
        """
        switch swing stage and leg group (if necessary)
        """
        self.stage = next(self.stages_cycle)
        if self.stage == StageType.UP:
            self.legs = next(self.legs_cycle)
        return self.legs, self.stage

    @abstractmethod
    def get_legs_cycle(self):
        pass


class _3LegCycle(_Cycle):
    """
    tripod
    """

    def get_legs_cycle(self):
        return cycle([
            [leg_rf, leg_rr, leg_lm],
            [leg_lf, leg_lr, leg_rm]
        ])


class _1LegCycle(_Cycle):
    """
    wave
    """

    def get_legs_cycle(self):
        return cycle([
            [leg_lr],
            [leg_rf],
            [leg_rm],
            [leg_rr],
            [leg_lf],
            [leg_lm]
        ])


class _2LegCycle(_Cycle):
    """
    ripple
    """

    def get_legs_cycle(self):
        return cycle([
            [leg_rm, leg_lr],
            [leg_rf, leg_lm],
            [leg_lf, leg_rr]
        ])
