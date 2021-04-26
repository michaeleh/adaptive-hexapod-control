import numpy as np
from abc import ABC, abstractmethod

from SNN.body_level_sensing import BodyOrientationModel
from environment.leg import Leg, leg_rf, leg_rr, leg_rm, leg_lm


class AbstractBodyOrientation(ABC):
    def __init__(self, env):
        self.env = env
        self.wide_anchors = [leg_rm, leg_lm]  # h = RM-LM
        self.long_anchors = [leg_rr, leg_rf]  # h = rr-rf

    @abstractmethod
    def get_theta_wide(self):
        """
        :return: current and target orientation
        """
        pass

    @abstractmethod
    def get_theta_long(self):
        """
        :return: current and target orientation
        """
        pass

    def get_theta(self, axis):
        assert axis in ['wide', 'long']
        if axis == 'wide':
            return self.get_theta_wide()
        if axis == 'long':
            return self.get_theta_long()

    def get_wide_points(self):
        p1 = self.env.get_pos(self.wide_anchors[0].coxa.value)[1:]
        p2 = self.env.get_pos(self.wide_anchors[1].coxa.value)[1:]
        return np.array([p1, p2])

    def get_long_points(self):
        p1 = self.env.get_pos(self.long_anchors[0].coxa.value)[[0, 2]]
        p2 = self.env.get_pos(self.long_anchors[1].coxa.value)[[0, 2]]
        return np.array([p1, p2])

    def wrap_angle_around_axis(self, leg: Leg, axis):
        """
        wrap angle around axis of orientation pi removing pi from it or not
        :param leg: leg or adjustment
        :param: axis or rotation x or y
        :return: new angle target
        """
        assert axis in ['wide', 'long']
        side, loc = leg.position()
        if axis == 'wide':
            wide_side, _ = self.wide_anchors[0].position()
            if side == wide_side:  # rear is 0 around x axis
                return 0
            return -np.pi
        if axis == 'long':
            _, long_loc = self.long_anchors[0].position()
            if loc == long_loc:  # right is 0 around y axis
                return 0
            return -np.pi


class SimBodyOrientation(AbstractBodyOrientation):

    def get_theta_wide(self):
        p1, p2 = self.get_wide_points()
        w, h = (p1 - p2)
        theta = np.arctan2(h, w)
        return theta

    def get_theta_long(self):
        p1, p2 = self.get_long_points()
        w, h = (p1 - p2)
        theta = np.arctan2(h, w)
        return theta


class NeuromorphicOrientationModel(AbstractBodyOrientation):
    def __init__(self, env):
        super().__init__(env)
        self.model = BodyOrientationModel(wide0_points=self.get_wide_points(),
                                          long0_points=self.get_long_points(),
                                          frame_skip=env.frame_skip, debug=True)
        self.prev_wideh = self.get_wide_points().T[1]
        self.prev_longh = self.get_long_points().T[1]
        self.history = []

    def get_theta_long(self):
        _, long = self.model.curr_val
        return long

    def get_theta_wide(self):
        wide, _ = self.model.curr_val
        return wide

    def update(self):
        new_wideh = self.get_wide_points().T[1]
        new_longh = self.get_long_points().T[1]

        wide_change = new_wideh - self.prev_wideh
        long_change = new_longh - self.prev_longh
        self.history.append(wide_change)
        self.model.update(wide_change, long_change)
        self.prev_wideh = new_wideh
        self.prev_longh = new_longh
