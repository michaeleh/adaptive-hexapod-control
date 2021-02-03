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
prevp1 = env.get_pos(joint1)
prevp2 = env.get_pos(joint2)
neuro_model = PlaneRotation(sim_dt=env.dt, dist=(prevp1[0] - prevp2[0]))
# warmup physic
for _ in range(1200):
    env.step(env.qpos, render=False)

prevp1 = prevp2 = np.zeros(3)
state = env.qpos
for v in qpos_map.values():
    state[v] = 0
env.set_state(state, np.zeros_like(env.qvel))
for _ in tqdm(range(1000)):
    env.step(env.qpos, render=False)
    p1 = env.get_pos(joint1)
    p2 = env.get_pos(joint2)
    neuro_model.update((p1[2] - prevp1[2]),
                       (p2[2] - prevp2[2]))
    prevp1, prevp2 = p1.copy(), p2.copy()

env.close()

