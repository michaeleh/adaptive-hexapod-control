import os

import numpy as np

from hexapod_env import HexapodEnv
from model import Model

BASE_DIR = os.path.dirname(__file__)

model_name = 'mk3'
xml_path = os.path.join(BASE_DIR, f'hexapod-models/{model_name}/{model_name}.xml')
env = HexapodEnv(xml_path, frame_skip=1)
space_size = 20
model = Model(joint_pos_dict=env.map_joint_pos())
obs = env.reset()
while True:
    action = model.generate_action(obs)
    state_space = np.linspace(obs, action, space_size)
    for state in state_space:
        obs, reward, done, info = env.step(state)
        env.render()
