import pickle

import numpy as np
import os

from tqdm import tqdm

from gait.body_leveling.body_orientation import NeuromorphicOrientationModel, SimBodyOrientation
from environment.hexapod_env import HexapodEnv
from gait.body_leveling.leveling_action import calculate_body_leveling_action

'''
Loading environment and environment
'''
BASE_DIR = os.path.dirname(__file__)
model_name = 'box'
xml_path = os.path.join(BASE_DIR, f'../mjcf_models/{model_name}.xml')
env = HexapodEnv(xml_path, frame_skip=300)

'''
Init gaits and other simulation variables
'''
qpos_map = env.map_joint_qpos
obs = env.reset()

pos = env.qpos * 0
pos[:3] = [-1, -0., -0.1]
env.set_state(pos, 0 * env.qvel)

sim_model = SimBodyOrientation(env)
for i in range(2):
    env.step({}, render=True)  # warmup

# init
orientation_model = NeuromorphicOrientationModel(env)
env.step({}, orientation_model.update, render=True)

for i in tqdm(range(3)):
    action = calculate_body_leveling_action(orientation_model, env.qpos, qpos_map, 'x')
    # calculate the rotation change
    obs, reward, done, info = env.step(action, callback=orientation_model.update, render=True)
    action = calculate_body_leveling_action(orientation_model, env.qpos, qpos_map, 'y')
    # calculate the rotation change
    obs, reward, done, info = env.step(action, callback=orientation_model.update, render=True)
    orientation_model.model.save_figs(axis='x')
    orientation_model.model.save_figs(axis='y')

with open('tmp.pkl', 'wb') as fp:
    pickle.dump(orientation_model.history, fp)
