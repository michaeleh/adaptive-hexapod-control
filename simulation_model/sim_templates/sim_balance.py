from scipy.spatial.transform import Rotation
from scipy.constants import g
from simulation_model.sim_templates.abstract_sim_template import AbstractSimTemplate
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import nearest_points
import numpy as np

MM_IN_METER = 1000


class SimBalance(AbstractSimTemplate):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt
        self.vz = 0
        self.angular_vel = np.zeros(2)
        self.t = 0

    def reset(self):
        self.vz = 0
        self.angular_vel = np.zeros(2)
        self.t = 0

    def eval(self, curr_pos, contact_pos, Lx, Ly):
        """
        move orientation and position in respect ot center of mass
        :param Ly: length in y axis of body
        :param Lx: length in x axis of body
        :param contact_pos: positions of legs supporting
        :param curr_pos: current position of hexapod
        :return: rotation axis due to balance
        """
        # distance of COM from base (supporting polygon)
        self.t += self.dt  # progress time
        COM = curr_pos[:3]  # center of mass is center of hexapod
        p_com = Point(COM[:2])  # projected on 2d
        base = [p[:2] for p in contact_pos]  # projected on 2d
        if len(base) < 3:  # no support polygon
            self.vz = 0.5 * g * self.dt * self.t ** 2  # free fall gravity in mm (not meters)
            curr_pos[2] -= self.vz  # free fall in Z
            return curr_pos
        supporting_polygon = Polygon(base)
        # if com in supporting polygon, all is good
        if supporting_polygon.contains(p_com):
            self.reset()
            return curr_pos

        p1, p2 = nearest_points(supporting_polygon, p_com)  # p1 is the polygon point
        x, y = p1.xy
        x = x[0]
        y = y[0]
        diff_direction = [COM[0] - x, COM[1] - y]
        # if we tip towards x we need to rotate around y and vise versa
        # angle of rotation is atctn of height and the distance
        anglex = np.arctan2(diff_direction[1], COM[2])
        angley = np.arctan2(diff_direction[0], COM[2])  # angle between normal and current position of angle
        L = np.array([Lx, Ly])
        theta = np.array([anglex, angley])
        self.calc_v_theta(L, theta)
        self.angular_fall(curr_pos)

        return curr_pos

    def angular_fall(self, curr_pos):
        """
        update free fall object position
        :param curr_pos:
        :return:
        """
        w, x, y, z = curr_pos[3:7]  # current orientation
        q = Rotation.from_quat([x, y, z, w])
        curr_rot = q.as_euler('xyz', degrees=False)
        # update rotation
        vtheta = np.array([*self.angular_vel, 0])  # expanding to 3d
        rot = Rotation.from_euler('xyz', curr_rot + vtheta * self.dt, degrees=False)
        x, y, z, w = rot.as_quat()
        curr_pos[3:7] = [w, x, y, z]

    def calc_v_theta(self, L, angle):
        """
        angular velocity calculation from
        https://www.wired.com/2014/09/how-long-does-it-take-for-a-pencil-to-tip-over/
        and based on the invert pendulum principle
        """
        end_theta = np.pi / 2
        angle[angle <= end_theta] = 0  # no change if bigger than 90 deg
        angular_acc = 3 * np.sin(angle) * g / (2 * L)
        self.angular_vel += angular_acc * self.dt
