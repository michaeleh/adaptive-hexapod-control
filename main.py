import os
import time
from itertools import cycle

import numpy as np
from numpy import linspace

from gait.motion import TripodMotion, WaveMotion, RippleMotion
from simulation_model.hexapod_env import HexapodEnv

'''
Loading simulation_model and environment
'''
BASE_DIR = os.path.dirname(__file__)
model_name = 'mk3'
xml_path = os.path.join(BASE_DIR, f'mujoco-models/{model_name}/{model_name}.xml')
env = HexapodEnv(xml_path, frame_skip=10)

'''
Init gaits and other simulation variables
'''
qpos_map = env.map_joint_qpos
models = cycle([WaveMotion(qpos_map), RippleMotion(qpos_map), TripodMotion(qpos_map)])
obs = env.reset()
space_size = 10  # how many state to interpolate
model = next(models)
start_time = time.time()
# for _ in range(1000):
#     env.sim.step()
# obs = env.get_obs()
while True:

    curr_time = time.time()
    # reset and change simulation_model
    if curr_time - start_time > 10:  # 8 seconds passed
        model = next(models)
        start_time = curr_time
        env.reset_model()
        goal = 0  # reset simulation_model
        obs, reward, done, info = env.step(np.zeros_like(env.qpos))
    else:
        # get simulation_model action
        goal = model.generate_action(obs)
        # interpolate
    for state in linspace(env.get_obs(), goal, space_size):
        obs, reward, done, info = env.step(state, render=True)
env.close()
