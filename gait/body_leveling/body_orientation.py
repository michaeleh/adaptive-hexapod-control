import numpy as np
from abc import ABC, abstractmethod

from environment.leg import Leg, leg_rf, leg_rr, leg_rm, leg_lm


class AbstractBodyOrientation(ABC):
    def __init__(self):
        self.x_anchors = [leg_rm, leg_lm]  # h = RM-LM
        self.y_anchors = [leg_rr, leg_rf]  # h = rr-rf

    @abstractmethod
    def get_theta_x(self):
        """
        :return: current and target orientation
        """
        pass

    @abstractmethod
    def get_theta_y(self):
        """
        :return: current and target orientation
        """
        pass

    def get_theta(self, axis):
        assert axis in ['x', 'y']
        if axis == 'x':
            return self.get_theta_x()
        if axis == 'y':
            return self.get_theta_y()

    def wrap_angle_around_axis(self, leg: Leg, axis):
        """
        wrap angle around axis of orientation pi removing pi from it or not
        :param leg: leg or adjustment
        :param: axis or rotation x or y
        :return: new angle target
        """
        assert axis in ['x', 'y']
        side, loc = leg.position()
        if axis == 'x':
            x_side, _ = self.x_anchors[0].position()
            if side == x_side:  # rear is 0 around x axis
                return 0
            return -np.pi
        if axis == 'y':
            _, y_loc = self.y_anchors[0].position()
            if loc == y_loc:  # right is 0 around y axis
                return 0
            return -np.pi


class SimBodyOrientation(AbstractBodyOrientation):
    def __init__(self, env):
        super().__init__()
        self.env = env

    def get_theta_x(self):
        p1 = self.env.get_pos(self.x_anchors[0].coxa.value)
        p2 = self.env.get_pos(self.x_anchors[1].coxa.value)
        w, h = (p1 - p2)[1:]  # y and z
        theta = np.arctan2(h, w)
        target_theta = 0
        return theta, target_theta

    def get_theta_y(self):
        p1 = self.env.get_pos(self.y_anchors[0].coxa.value)
        p2 = self.env.get_pos(self.y_anchors[1].coxa.value)
        d = (p1 - p2)
        w, h = d[0], d[2]  # x and z
        theta = np.arctan2(h, w)
        target_theta = 0
        return theta, target_theta
