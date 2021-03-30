import numpy as np
import os

from gait.body_leveling.body_orientation import SimBodyOrientation
from gait.body_leveling.leveling_action import calculate_body_leveling_action
from gait.gait_impl import TripodMotion
from environment.hexapod_env import HexapodEnv
from kinematics.ik_algorithm import angles_to_target
from kinematics.joint_kinematics import KinematicNumericImpl

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
pos[:3] = [-1, -0, -0.1]
env.set_state(pos, 0 * env.qvel)

while True:
    action = calculate_body_leveling_action(SimBodyOrientation(env), env.qpos, qpos_map, 'x')
    # calculate the rotation change
    obs, reward, done, info = env.step(action, render=True)
    # env.set_state(qpos, 0 * env.qvel)
    # i+=1
    # if i <1000:
    #     env.sim.step()
    # env.render()
