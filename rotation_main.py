import time

import matplotlib.pyplot as plt
import os

from numpy import linspace

from gait.motion import RotationMotion
from model.hexapod_env import HexapodEnv
from neuro.neuro_model import NeuroIntegrator

'''
Loading model and environment
'''
BASE_DIR = os.path.dirname(__file__)
model_name = 'mk3'
xml_path = os.path.join(BASE_DIR, f'mujoco-models/{model_name}/{model_name}.xml')
env = HexapodEnv(xml_path, frame_skip=1)

'''
Init gaits and other simulation variables
'''
qpos_map = env.map_joint_qpos()

motion_model = RotationMotion(qpos_map)
obs = env.reset()
space_size = 100  # how many state to interpolate

integrator = NeuroIntegrator(env.dt)
h = env.hexa_h()
history = []

for _ in range(15):
    # get model action
    goal = motion_model.generate_action(obs, env.axis_change())
    # interpolate
    for state in linspace(env.get_obs(), goal, space_size):
        obs, reward, done, info = env.step(state)
        env.render()
        new_h = env.hexa_h()
        history.append(new_h)
        integrator.update(new_h - h)
        h = new_h
    start = time.time()
    # dont move
    while time.time() - start < 1:
        obs, reward, done, info = env.step(state)
        env.render()
        new_h = env.hexa_h()
        history.append(new_h)
        integrator.update(new_h - h)
        h = new_h

env.close()
plt.title('Integrator estimation of hexapod heights')
plt.plot(*integrator.get_xy(), label='intergrator')
plt.plot(integrator.get_xy()[0], history, linestyle='--', label='real height')
plt.legend()
plt.grid()
plt.savefig('neurotrophic_height_dont_move.png')
# plt.show()
