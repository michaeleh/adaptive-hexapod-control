from scipy.spatial.transform import Rotation
from scipy.constants import g
from simulation_model.sim_templates.abstract_sim_template import AbstractSimTemplate
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.ops import nearest_points
import numpy as np


class SimBalance(AbstractSimTemplate):
    def __init__(self, dt):
        super().__init__()
        self.dt = dt
        self.vz = 0
        self.vtheta = np.zeros(3)
        self.t = 0

    def reset(self):
        self.vz = 0  # 3d velocities
        self.vtheta = np.zeros(3)  # x and y angles
        self.t = 0

    def eval(self, curr_pos, contact_pos):
        """
        move orientation and position in respect ot center of mass
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
            self.vz = 100 * 0.5 * g * self.t ** 2  # free fall gravity
            self.free_fall(curr_pos)
            return curr_pos
        supporting_polygon = Polygon(base)
        # if com in supporting polygon, all is good
        if supporting_polygon.contains(p_com):
            self.reset()
            return curr_pos
        print(COM)
        print(base)
        print()
        exit()
        p1, p2 = nearest_points(supporting_polygon, p_com)  # p1 is the polygon point
        x, y = p1.xy
        x = x[0]
        y = y[0]
        diff_direction = [COM[0] - x, COM[1] - y]
        # if we tip towards x we need to rotate around y and vise versa
        # angle of rotation is atctn of height and the distance TODO research physical behaviour
        anglex = np.arctan2(COM[2], diff_direction[1]) * self.dt
        angley = np.arctan2(COM[2], diff_direction[0]) * self.dt
        self.vtheta = np.array([anglex, angley, 0])
        self.free_fall(curr_pos)  # TODO someting
        return curr_pos

    def free_fall(self, curr_pos):
        """
        update free fall object position
        :param curr_pos:
        :return:
        """
        curr_pos[2] -= self.vz  # free fall in Z
        w, x, y, z = curr_pos[3:7]  # current orientation
        q = Rotation.from_quat([x, y, z, w])
        curr_rot = q.as_euler('xyz', degrees=False)
        # update rotation
        rot = Rotation.from_euler('xyz', curr_rot - self.vtheta, degrees=False)
        x, y, z, w = rot.as_quat()
        curr_pos[3:7] = [w, x, y, z]
