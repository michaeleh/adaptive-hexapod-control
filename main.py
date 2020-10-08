import os

from numpy import linspace

from hexapod_env import HexapodEnv
from model import Model

BASE_DIR = os.path.dirname(__file__)
model_name = 'mk3'
xml_path = os.path.join(BASE_DIR, f'hexapod-models/{model_name}/{model_name}.xml')
env = HexapodEnv(xml_path, frame_skip=50)
model = Model(joint_pos_dict=env.map_joint_qpos())
obs = env.reset()
space_size = 10
while True:
    goal = model.generate_action(obs)
    for state in linspace(env.get_obs(), goal, space_size):
        obs, reward, done, info = env.step(state)
        env.render()
