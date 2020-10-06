import os

import numpy as np

from hexapod_env import HexapodEnv
from model import Model

BASE_DIR = os.path.dirname(__file__)

model_name = 'mk3'
xml_path = os.path.join(BASE_DIR, f'hexapod-models/{model_name}/{model_name}.xml')
frame_skip = 60
env = HexapodEnv(xml_path, frame_skip)

model = Model(joint_pos_dict=env.map_joint_pos())
obs = env.reset()
while True:
    action = model.generate_action(obs)
    state_space = np.linspace(obs, action, frame_skip)
    for state in state_space:
        obs, reward, done, info = env.step(state)
        env.render()
