import pickle

import matplotlib.pyplot as plt
import os

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from kinematics.constants import DeltaLengths
from kinematics.ik_algorithm import angles_to_target
from model.hexapod_env import HexapodEnv
from model.joint_types import JointNames
from model.leg import all_legs
from neuro.plane_angles import PlaneRotation

'''
Loading model and environment
'''
BASE_DIR = os.path.dirname(__file__)
xml_path = os.path.join(BASE_DIR, 'mujoco-models/mk3/mk3_body_level.xml')
env = HexapodEnv(xml_path, frame_skip=1)  # frame skip should match model dt so keep frame_skip=1
qpos_map = env.map_joint_qpos()
obs = env.reset()
space_size = 10  # how many state to interpolate

# y axis angle joints
joint1 = JointNames.COXA_RM.value
joint2 = JointNames.COXA_LM.value

# model
neuro_model = PlaneRotation(sim_dt=env.dt)
prevp1 = prevp2 = np.zeros(3)  # prev pos init to 0

x_length = DeltaLengths.COXA_FEMUR_X + DeltaLengths.FEMUR_TIBIA_X + DeltaLengths.TIBIA_END_X
y_length = DeltaLengths.COXA_FEMUR_Y + DeltaLengths.FEMUR_TIBIA_Y + DeltaLengths.TIBIA_END_Y
z_length = DeltaLengths.COXA_FEMUR_Z + DeltaLengths.FEMUR_TIBIA_Z + DeltaLengths.TIBIA_END_Z


def warmup():
    for _ in range(1200):
        env.step(env.qpos, render=False)


def zero_joints():
    state = env.qpos
    for v in qpos_map.values():
        state[v] = 0
    env.set_state(state, np.zeros_like(env.qvel))
    return state


def stim_model_with_update():
    global prevp1, prevp2
    p1 = env.get_pos(joint1)
    p2 = env.get_pos(joint2)
    idxs = [1, 2]  # x,z axis
    neuro_model.update((p1[idxs] - prevp1[idxs]),
                       (p2[idxs] - prevp2[idxs]))
    prevp1, prevp2 = p1.copy(), p2.copy()


def rotate_around_origin():
    # set orientation
    new_rot = env.curr_rot - np.array([0, rot_angle, 0])  # only y axis rotation
    rot = Rotation.from_euler('xyz', new_rot, degrees=False)
    x, y, z, w = rot.as_quat()
    # set
    new_pos[3:7] = [w, x, y, z]
    new_pos[0:3] = zero_state[0:3]


def simulate(t):
    for _ in tqdm(range(t)):
        # update SNN
        stim_model_with_update()
        env.step(new_pos, render=True)


def cartesian_change(r, src, dst):
    x0 = r * np.cos(src)
    y0 = r * np.sin(src)
    x1 = r * np.cos(dst)
    y1 = r * np.sin(dst)
    return x1 - x0, y1 - y0


def adjust_leg_placements():
    real_angle = env.curr_rot[1]
    for leg in all_legs:
        # get joints
        coxa = qpos_map[leg.coxa.value]
        femur = qpos_map[leg.femur.value]
        tibia = qpos_map[leg.tibia.value]
        joint_pos = [coxa, femur, tibia]

        # side of leg effect y axis side a or (a - pi)
        side, _ = leg.position()
        src = rot_angle if side == 'R' else rot_angle - np.pi
        dst = real_angle - rot_angle if side == 'R' else real_angle - rot_angle - np.pi

        # y axis rotation
        r = leg.rotate([x_length, y_length, z_length])
        r = np.sqrt(r[1] ** 2 + r[2] ** 2)
        x_change, z_change = cartesian_change(r, src, dst)  # round for smooth
        change = np.array([x_change, 0, z_change])
        new_pos[joint_pos], e = angles_to_target(q=env.qpos[joint_pos], target=leg.rotate(change))


if __name__ == '__main__':
    # warmup physic
    warmup()
    # set position to all joint 0
    zero_state = zero_joints()

    new_pos = env.qpos
    simulate(1000)

    rot_angle = neuro_model.curr_val
    rotate_around_origin()
    adjust_leg_placements()
    simulate(1000)
    env.close()
