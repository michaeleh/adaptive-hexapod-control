import numpy as np
from abc import ABC, abstractmethod

from SNN.body_level_sensing import BodyOrientationModel
from environment.leg import Leg, leg_rf, leg_rr, leg_rm, leg_lm


class AbstractBodyOrientation(ABC):
    def __init__(self, env):
        self.env = env
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

    def get_x_points(self):
        p1 = self.env.get_pos(self.x_anchors[0].coxa.value)[1:]
        p2 = self.env.get_pos(self.x_anchors[1].coxa.value)[1:]
        return np.array([p1, p2])

    def get_y_points(self):
        p1 = self.env.get_pos(self.y_anchors[0].coxa.value)[[0, 2]]
        p2 = self.env.get_pos(self.y_anchors[1].coxa.value)[[0, 2]]
        return np.array([p1, p2])

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

    def get_theta_x(self):
        p1, p2 = self.get_x_points()
        w, h = (p1 - p2)
        theta = np.arctan2(h, w)
        return theta

    def get_theta_y(self):
        p1, p2 = self.get_y_points()
        w, h = (p1 - p2)
        theta = np.arctan2(h, w)
        return theta


class NeuromorphicOrientationModel(AbstractBodyOrientation):
    def __init__(self, env):
        super().__init__(env)
        self.model = BodyOrientationModel(x0_points=self.get_x_points(),
                                          y0_points=self.get_y_points(),
                                          sim_dt=env.dt, frame_skip=env.frame_skip)
        self.prev_xh = self.get_x_points().T[1]
        self.prev_yh = self.get_y_points().T[1]
        self.history = []

    def get_theta_y(self):
        _, y = self.model.curr_val
        return y

    def get_theta_x(self):
        x, _ = self.model.curr_val
        return x

    def update(self):
        new_xh = self.get_x_points().T[1]
        new_yh = self.get_y_points().T[1]

        x_change = new_xh - self.prev_xh
        y_change = new_yh - self.prev_yh
        self.history.append(x_change)
        self.model.update(x_change, y_change)
        self.prev_xh = new_xh
        self.prev_yh = new_yh
