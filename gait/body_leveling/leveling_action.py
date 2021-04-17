import numpy as np
from environment.leg import all_legs
from gait.body_leveling.body_orientation import AbstractBodyOrientation
from kinematics.ik_algorithm import angles_to_target
from kinematics.joint_kinematics import KinematicNumericImpl
from utils.coordinates import cartesian_change

fk = KinematicNumericImpl()


def extend(diff, axis):
    assert axis in ['x', 'y']
    if axis == 'x':
        return np.array([0, *diff])
    if axis == 'y':
        return np.array([diff[0], 0, diff[1]])


def calcualte_r(angles, leg, axis):
    assert axis in ['x', 'y']
    xyz = fk.calc_xyz(angles)
    xyz = leg.rotate(xyz)
    if axis == 'x':
        r = xyz[1:]
    if axis == 'y':
        r = [xyz[0], xyz[2]]
    return np.linalg.norm(r)


def calculate_body_leveling_action(body_or: AbstractBodyOrientation, qpos, qpos_map, axis):
    assert axis in ['x', 'y']
    theta = body_or.get_theta(axis)
    action = {}
    for leg in all_legs:
        coxa = qpos_map[leg.coxa.value]
        femur = qpos_map[leg.femur.value]
        tibia = qpos_map[leg.tibia.value]
        joint_pos = [coxa, femur, tibia]
        angles = qpos[joint_pos]

        r = calcualte_r(angles, leg, axis)
        src = theta + body_or.wrap_angle_around_axis(leg, axis)  # -pi or not
        dst = 0 + body_or.wrap_angle_around_axis(leg, axis)

        diff = cartesian_change(r, src, dst)
        diff = extend(diff, axis)
        q, _ = angles_to_target(angles, -diff)
        action[leg.coxa.value], action[leg.femur.value], action[leg.tibia.value] = q

    return action
