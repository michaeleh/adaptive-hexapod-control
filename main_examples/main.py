import os
import time
from itertools import cycle

from gait.gait_impl import TripodMotion, WaveMotion, RippleMotion
from environment.hexapod_env import HexapodEnv

'''
Loading environment and environment
'''
BASE_DIR = os.path.dirname(__file__)
model_name = 'hexapod'
xml_path = os.path.join(BASE_DIR, f'../mjcf_model/{model_name}.xml')
env = HexapodEnv(xml_path, frame_skip=500)

'''
Init gaits and other simulation variables
'''
qpos_map = env.map_joint_qpos
gaits = cycle([WaveMotion(qpos_map), RippleMotion(qpos_map), TripodMotion(qpos_map)])
obs = env.reset()
gait = next(gaits)
i = 0
start_time = time.time()
while True:

    curr_time = time.time()
    # reset and change environment
    if curr_time - start_time > 10:
        gait = next(gaits)
        start_time = curr_time
        obs = env.reset_model()
        goal = 0  # reset environment
    else:
        # get environment action
        action = gait.generate_action(obs)
        obs, reward, done, info = env.step(action, render=True)
