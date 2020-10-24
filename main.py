import os
import time
from itertools import cycle
import numpy as np
from numpy import linspace

from gait.motion import TripodMotion, WaveMotion, RippleMotion
from model.hexapod_env import HexapodEnv

BASE_DIR = os.path.dirname(__file__)
model_name = 'mk3'
xml_path = os.path.join(BASE_DIR, f'mujoco-models/{model_name}/{model_name}.xml')
env = HexapodEnv(xml_path, frame_skip=10)

qpos_map = env.map_joint_qpos()
models = cycle([WaveMotion(qpos_map), RippleMotion(qpos_map), TripodMotion(qpos_map)])
obs = env.reset()
space_size = 10
model = next(models)
start_time = time.time()
while True:
    curr_time = time.time()
    # reset and change model
    if curr_time - start_time > 10:  # 8 seconds passed
        model = next(models)
        start_time = curr_time
        env.set_state(np.zeros_like(env.qpos), np.zeros_like(env.qvel))
        goal = 0
        obs, reward, done, info = env.step(np.zeros_like(env.qpos))
    else:
        # get model action
        goal = model.generate_action(obs)
    for state in linspace(env.get_obs(), goal, space_size):
        obs, reward, done, info = env.step(state)
