from abc import ABC, abstractmethod

from environment.leg import leg_rf, leg_rm, leg_rr, leg_lf, leg_lm, leg_lr
from gait.state_transitions import StateTransitions


class _Cycle(ABC):
    """
    Defining leg groups for gaits cycle.
    """

    def __init__(self):
        self.stages_cycle = StateTransitions(self.get_legs())

    def get_next(self):
        """
        switch swing stage and leg group (if necessary)
        """
        return next(self.stages_cycle)

    @abstractmethod
    def get_legs(self):
        pass


class _3LegCycle(_Cycle):
    """
    tripod
    """

    def get_legs(self):
        return [
            [leg_rf, leg_rr, leg_lm],
            [leg_lf, leg_lr, leg_rm]
        ]


class _1LegCycle(_Cycle):
    """
    wave
    """

    def get_legs(self):
        return [
            [leg_lr],
            [leg_rf],
            [leg_rm],
            [leg_rr],
            [leg_lf],
            [leg_lm]
        ]


class _2LegCycle(_Cycle):
    """
    ripple
    """

    def get_legs(self):
        return [
            [leg_rm, leg_lr],
            [leg_rf, leg_lm],
            [leg_lf, leg_rr]
        ]
