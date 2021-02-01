import os
import time
from itertools import cycle

import numpy as np
from numpy import linspace
from scipy.spatial.transform import Rotation

from gait.motion import TripodMotion, WaveMotion, RippleMotion
from model.hexapod_env import HexapodEnv

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

# warmup physic
for _ in range(100):
    env.step(env.qpos, render=False)

state = env.qpos
for v in qpos_map.values():
    state[v] = 0
env.set_state(state, np.zeros_like(env.qvel))
while True:
    env.step(env.qpos, render=True)
env.close()
