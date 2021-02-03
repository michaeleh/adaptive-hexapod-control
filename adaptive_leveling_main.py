import pickle

import matplotlib.pyplot as plt
import os

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from model.hexapod_env import HexapodEnv
from model.joint_types import JointNames
from neuro.plane_angles import PlaneRotation
from neuro.utils import SCALE

'''
Loading model and environment
'''
BASE_DIR = os.path.dirname(__file__)
xml_path = os.path.join(BASE_DIR, 'mujoco-models/mk3/mk3_body_level.xml')
env = HexapodEnv(xml_path, frame_skip=1)

'''
Init gaits and other simulation variables
'''
qpos_map = env.map_joint_qpos()
obs = env.reset()
space_size = 10  # how many state to interpolate

joint1 = JointNames.COXA_RM.value
joint2 = JointNames.COXA_LM.value

# warmup physic
for _ in range(1200):
    env.step(env.qpos, render=False)

neuro_model = PlaneRotation(sim_dt=env.dt)

prevp1 = prevp2 = np.zeros(3)
state = env.qpos
for v in qpos_map.values():
    state[v] = 0
env.set_state(state, np.zeros_like(env.qvel))
history1 = []
history2 = []
for _ in tqdm(range(1000)):
    env.step(env.qpos, render=False)
    p1 = env.get_pos(joint1)
    p2 = env.get_pos(joint2)
    idxs = [0, 2]  # x,z axis
    neuro_model.update((p1[idxs] - prevp1[idxs]),
                       (p2[idxs] - prevp2[idxs]))
    prevp1, prevp2 = p1.copy(), p2.copy()
    history1.append(p1[idxs])
    history2.append(p2[idxs])
env.close()

x, y1, y2 = neuro_model.get_xy()
plt.plot(x, y1, label='1', c='g')
plt.plot(x, y2, label='2', c='b')
plt.plot(x, history1, label='real1', linestyle='--', c='g')
plt.plot(x, history2, label='real2', linestyle='--', c='b')
plt.legend()
plt.savefig('gihhg.png')
