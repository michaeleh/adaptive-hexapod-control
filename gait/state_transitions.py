from enum import Enum


class StageType(Enum):
    UP, FORWARD, DOWN, RETURN, LEVEL = range(5)


class _Index:
    def __init__(self, init_val, max_val):
        assert init_val < max_val
        self.max_val = max_val
        self.init_val = init_val
        self.idx = 0

    def reset(self):
        self.idx = self.init_val

    def increment(self):
        self.idx += 1
        if self.idx == self.max_val:
            self.reset()

    def wrap(self, v):
        if v >= self.max_val:
            v = (v % self.max_val) + self.init_val
        return v


class StateTransitions(object):
    # state tuple for current leg group and the next one
    states = [
        (StageType.UP, None),
        (StageType.FORWARD, None),
        (StageType.DOWN, None),
        (StageType.LEVEL, StageType.LEVEL),
        (None, StageType.UP),
        (StageType.RETURN, StageType.FORWARD),

    ]

    def __init__(self, legs):
        self.legs = legs
        self.reset()

    def reset(self):
        self.idx_leg = _Index(init_val=0, max_val=len(self.legs))
        self.idx_state = _Index(init_val=2, max_val=len(self.states))

    def __next__(self):
        states = self.states[self.idx_state.idx]
        ret = self.curr
        self.idx_state.increment()
        if states == (StageType.RETURN, StageType.FORWARD):
            self.idx_leg.increment()
        return ret

    @property
    def curr(self):
        states = self.states[self.idx_state.idx]
        leg_i = self.idx_leg.idx
        legs1 = self.legs[leg_i]
        legs2 = self.legs[self.idx_leg.wrap(leg_i + 1)]
        return dict(zip(states, (legs1, legs2)))
