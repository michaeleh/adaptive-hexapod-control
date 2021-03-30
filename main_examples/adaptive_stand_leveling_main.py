import numpy as np
import os

from environment.leg import all_legs, Side
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

fk = KinematicNumericImpl()


def cartesian_change(r, src, dst):
    x0 = r * np.cos(src)
    y0 = r * np.sin(src)
    x1 = r * np.cos(dst)
    y1 = r * np.sin(dst)
    return np.array([0, x1 - x0, y1 - y0])  # y,z plane


i = 0
while True:
    # get angle of rotation
    rm = env.get_pos('coxa_RM')
    lm = env.get_pos('coxa_LM')
    w, h = (rm - lm)[1:]  # y and z
    theta = np.arctan2(h, w)
    target_theta = 0

    # calculate the rotation change
    action = {}
    qpos = env.qpos
    for leg in all_legs:
        coxa = qpos_map[leg.coxa.value]
        femur = qpos_map[leg.femur.value]
        tibia = qpos_map[leg.tibia.value]
        joint_pos = [coxa, femur, tibia]
        angles = env.qpos[joint_pos]

        r = np.linalg.norm(fk.calc_xyz(angles)[1:])
        if leg.side == Side.R:
            src = theta
            dst = target_theta

        if leg.side == Side.L:
            src = theta - np.pi
            dst = target_theta - np.pi

        diff = cartesian_change(r, src, dst)
        q, _ = angles_to_target(angles, -diff)
        # qpos[joint_pos] = q
        action[leg.coxa.value], action[leg.femur.value], action[leg.tibia.value] = q
        # fix rotatation angles
    obs, reward, done, info = env.step(action, render=True)
    # env.set_state(qpos, 0 * env.qvel)
    # i+=1
    # if i <1000:
    #     env.sim.step()
    # env.render()
