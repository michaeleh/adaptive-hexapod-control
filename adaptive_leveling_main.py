import matplotlib.pyplot as plt
import os

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from kinematics.ik_algorithm import angles_to_target
from kinematics.joint_kinematics import KinematicNumericImpl
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
    new_pos = env.qpos
    rot = Rotation.from_euler('xyz', [0, -rot_angle, 0], degrees=True)
    x, y, z, w = rot.as_quat()
    new_pos[3:7] = [w, x, y, z]
    env.set_state(new_pos, 0 * env.qvel)
    env.render()
    # adjust position
    # adjustment_pos = env.qpos
    # adjustment_pos[0:3] = np.zeros(3)
    # for leg in all_legs:
    #     coxa = qpos_map[leg.coxa.value]
    #     femur = qpos_map[leg.femur.value]
    #     tibia = qpos_map[leg.tibia.value]
    #     joint_pos = [coxa, femur, tibia]
    #     adjustment_pos[joint_pos], e = angles_to_target(q=adjustment_pos[joint_pos], target=-leg.rotate(diff))
    #
    # env.set_state(adjustment_pos, 0 * env.qvel)
    # env.render()
env.close()

# x, y = neuro_model.get_xy()
# plt.plot(x, y, label='model')
# plt.plot(x, history, label='real', linestyle='--')
# # plt.plot(x, history, label='real', linestyle='--')
# plt.legend()
# plt.savefig('gihhg.png')
