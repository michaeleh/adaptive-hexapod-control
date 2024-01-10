import pickle
from time import sleep
import numpy as np
import os

from tqdm import tqdm

from environment.hexapod_env import HexapodEnv
from gait.body_leveling.legs_heights import SimLegHeightModel, NeuromorphicLegHeightModel
from gait.body_leveling.leveling_action import calculate_body_leveling_action

'''
Loading environment and environment
'''
BASE_DIR = os.path.dirname(__file__)
model_name = 'ramp'
xml_path = os.path.join(BASE_DIR, f'../mjcf_models/{model_name}.xml')
env = HexapodEnv(xml_path, frame_skip=300)

'''
Init gaits and other simulation variables
'''
qpos_map = env.map_joint_qpos
obs = env.reset()

end = [-2.8, 0, 0]
pos2 = [-2.1, 0, 0]
pos1 = [-.5, 0, 0]
middle = [-1.28, 0, 0]
start = [.3, 0, 0]
state = middle

sim_model = SimLegHeightModel(env)
pos = env.qpos * 0
pos[:3] = state  # x axis leveling
env.set_state(pos, 0 * env.qvel)

for i in range(2):
    env.step({}, render=True)  # warmup

# init
orientation_model = NeuromorphicLegHeightModel(env)
env.step({}, callback=orientation_model.update, render=True)

for i in tqdm(range(5)):
    action = calculate_body_leveling_action(orientation_model, env.qpos, qpos_map)

    obs, reward, done, info = env.step(action, callback=orientation_model.update, render=True)

    orientation_model.model.save_figs()
