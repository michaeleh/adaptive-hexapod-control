import numpy as np
from environment.leg import all_legs
from gait.body_leveling.body_orientation import AbstractBodyOrientation
from gait.body_leveling.legs_heights import AbstractLegHeightModel
from kinematics.ik_algorithm import angles_to_target
from kinematics.joint_kinematics import KinematicNumericImpl
from utils.coordinates import cartesian_change
from utils import axis

fk = KinematicNumericImpl()


def extend(diff, axis):
    assert axis in ['wide', 'long']
    if axis == 'long':
        return np.array([0, *diff])
    if axis == 'wide':
        return np.array([diff[0], 0, diff[1]])


def calculate_r(angles, leg, axis):
    assert axis in ['wide', 'long']
    xyz = fk.calc_xyz(angles)
    xyz = leg.rotate(xyz)
    if axis == 'wide':
        r = xyz[1:]
    if axis == 'long':
        r = [xyz[0], xyz[2]]
    return np.linalg.norm(r)


def calculate_orientation_action(body_orientation: AbstractBodyOrientation, qpos, qpos_map, axis):
    assert axis in ['wide', 'long']
    theta = body_orientation.get_theta(axis)
    action = {}
    for leg in all_legs:
        coxa = qpos_map[leg.coxa.value]
        femur = qpos_map[leg.femur.value]
        tibia = qpos_map[leg.tibia.value]
        joint_pos = [coxa, femur, tibia]
        angles = qpos[joint_pos]

        r = calculate_r(angles, leg, axis)
        src = theta + body_orientation.wrap_angle_around_axis(leg, axis)  # -pi or not
        dst = 0 + body_orientation.wrap_angle_around_axis(leg, axis)

        diff = cartesian_change(r, src, dst)
        diff = extend(diff, axis)
        q, _ = angles_to_target(angles, diff)
        if np.count_nonzero(np.abs(q) > np.deg2rad(40)) > 0:  # if impossible then leave it
            continue
        action[leg.coxa.value], action[leg.femur.value], action[leg.tibia.value] = q

    return action, theta


def calculate_body_leveling_action(model: AbstractLegHeightModel, qpos, qpos_map):
    hs = model.get_legs_hs()
    target_height = hs.mean(axis=0)
    targets_list = target_height - hs

    action = {}
    for leg, target in zip(all_legs, targets_list):
        coxa = qpos_map[leg.coxa.value]
        femur = qpos_map[leg.femur.value]
        tibia = qpos_map[leg.tibia.value]
        joint_pos = [coxa, femur, tibia]
        angles = qpos[joint_pos]

        q, _ = angles_to_target(angles, target)
        action[leg.coxa.value], action[leg.femur.value], action[leg.tibia.value] = q

    return action
