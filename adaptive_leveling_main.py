import matplotlib.pyplot as plt
import os

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from kinematics.constants import DeltaLengths
from kinematics.ik_algorithm import angles_to_target
from kinematics.joint_kinematics import KinematicNumericImpl
from model.coordinates import cartesian_change
from model.hexapod_env import HexapodEnv
from model.joint_types import JointNames
from model.leg import all_legs
from neuro.plane_angles import PlaneRotation

'''
Loading model and environment
'''
BASE_DIR = os.path.dirname(__file__)
xml_path = os.path.join(BASE_DIR, 'mujoco-models/mk3/mk3_body_level.xml')
env = HexapodEnv(xml_path, frame_skip=10)

'''
Init gaits and other simulation variables
'''
qpos_map = env.map_joint_qpos()
obs = env.reset()
space_size = 10  # how many state to interpolate

joint1 = JointNames.COXA_RM.value
joint2 = JointNames.COXA_LM.value
ik_model = KinematicNumericImpl()
# warmup physic
for _ in range(1200):
    env.step(env.qpos, render=False)

neuro_model = PlaneRotation(sim_dt=env.dt)

x_length = DeltaLengths.COXA_FEMUR_X + DeltaLengths.FEMUR_TIBIA_X + DeltaLengths.TIBIA_END_X
y_length = DeltaLengths.COXA_FEMUR_Y + DeltaLengths.FEMUR_TIBIA_Y + DeltaLengths.TIBIA_END_Y
z_length = DeltaLengths.COXA_FEMUR_Z + DeltaLengths.FEMUR_TIBIA_Z + DeltaLengths.TIBIA_END_Z

prevp1 = prevp2 = np.zeros(3)
state = env.qpos
for v in qpos_map.values():
    state[v] = 0

env.set_state(state, np.zeros_like(env.qvel))
new_pos = state
for _ in tqdm(range(500)):
    p1 = env.get_pos(joint1)
    p2 = env.get_pos(joint2)
    idxs = [1, 2]  # x,z axis
    neuro_model.update((p1[idxs] - prevp1[idxs]),
                       (p2[idxs] - prevp2[idxs]))
    prevp1, prevp2 = p1.copy(), p2.copy()

    # rotate
    rot_angle = neuro_model.curr_val
    # set orientation
    qpos = env.qpos
    new_pos = qpos
    rot = Rotation.from_euler('xyz', [0, -rot_angle, 0], degrees=True)
    x, y, z, w = rot.as_quat()
    new_pos[3:7] = [w, x, y, z]
    new_pos[0:3] = [0, 0, 0]

    # adjust position
    for leg in all_legs:
        # get joints
        coxa = qpos_map[leg.coxa.value]
        femur = qpos_map[leg.femur.value]
        tibia = qpos_map[leg.tibia.value]
        joint_pos = [coxa, femur, tibia]
        # side of leg effect y axis side a or (a - pi)
        side, _ = leg.position()
        src = rot_angle if side == 'R' else rot_angle - np.pi
        dst = 0 if side == 'R' else -np.pi
        # y axis rotation
        r = leg.rotate([x_length, y_length, z_length])
        r = np.sqrt(r[0] ** 2 + r[2] ** 2)
        x_change, z_change = cartesian_change(r, src, dst)
        change = np.array([x_change, 0, z_change])
        new_pos[joint_pos], e = angles_to_target(q=qpos[joint_pos], target=leg.rotate(-change))

    env.step(new_pos, render=True)

env.close()

# x, y = neuro_model.get_xy()
# plt.plot(x, y, label='model')
# plt.plot(x, history, label='real', linestyle='--')
# # plt.plot(x, history, label='real', linestyle='--')
# plt.legend()
# plt.savefig('gihhg.png')
