import os

from hexapod_env import HexapodEnv
from model import Model

BASE_DIR = os.path.dirname(__file__)

model_name = 'mk3'
xml_path = os.path.join(BASE_DIR, f'hexapod-models/{model_name}/{model_name}.xml')
env = HexapodEnv(xml_path, 100)
model = Model(joint_pos_dict=env.map_joint_pos())
obs = env.reset()
while True:
    action = model.generate_action(obs)
    obs, reward, done, info = env.step(action)
    env.render()
