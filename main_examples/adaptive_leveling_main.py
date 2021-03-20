import os

from gait.gait_impl import TripodMotion
from environment.hexapod_env import HexapodEnv

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
gait = TripodMotion(qpos_map)
obs = env.reset()

while True:
    action = gait.generate_action(obs)
    obs, reward, done, info = env.step(action, render=True)
