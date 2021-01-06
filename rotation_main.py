import os
import time
from itertools import cycle

from numpy import linspace

from gait.motion import RotationMotion
from model.hexapod_env import HexapodEnv

'''
Loading model and environment
'''
BASE_DIR = os.path.dirname(__file__)
model_name = 'mk3'
xml_path = os.path.join(BASE_DIR, f'mujoco-models/{model_name}/{model_name}.xml')
env = HexapodEnv(xml_path, frame_skip=10)

'''
Init gaits and other simulation variables
'''
qpos_map = env.map_joint_qpos()

model = RotationMotion(qpos_map)
obs = env.reset()
space_size = 10  # how many state to interpolate

while True:
    # get model action
    goal = model.generate_action(obs, env.axis_change())
    # interpolate
    for state in linspace(env.get_obs(), goal, space_size):
        obs, reward, done, info = env.step(state)
