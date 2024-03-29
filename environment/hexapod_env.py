import os
import pathlib
import time

import matplotlib.pyplot as plt
from typing import Dict

import numpy as np
from gym.envs.mujoco import MujocoEnv

from environment.joint_types import JointNames
from utils.vectors import angle_between

np.set_printoptions(suppress=True)


class HexapodEnv(MujocoEnv):

    def __init__(self, model_path, frame_skip):
        super().__init__(model_path, frame_skip)
        self.dirname = ''
        self.direction = np.zeros(2)

    def reset_model(self):
        """
        reset environment to zero state
        """
        self.acuator_names = list(self.model.actuator_names)
        # self.save_ctrl()
        qpos = self.qpos * 0
        self.frames = []
        # fill_pos_defaults(qpos, self.map_joint_qpos)
        self.set_state(qpos, 0 * self.qvel)
        self.body_names = list(self.model.body_names)
        self.do_simulation(0 * self.ctrl, self.frame_skip)
        self.direction = 0.5 * (self.get_pos('coxa_RM') + self.get_pos('coxa_LM')) + np.array(
            [-10, 0, 0])  # proceed forward
        return self.get_obs()

    def step(self, action: Dict[str, float], callback=None, render=False, frame_skip=None):
        '''
        step simulation and implement action
        :param action: dict of joint_name -> desired angle
        :return: new observation
        '''

        info = dict()
        reward = 0
        done = False
        if frame_skip is None:
            frame_skip = self.frame_skip
        if isinstance(action, Dict):  # if step is action (or other mujoco's inner step runs)
            # Motion: pos after vanilla action
            ctrl = self.ctrl  # dont change current angles

            # replace changes values
            for joint, angle in action.items():
                idx = self.acuator_names.index(joint)
                ctrl[idx] = angle
            self.do_simulation(ctrl, frame_skip, callback, render)
            info['rad_to_target'] = self.rad_to_target()

        return self.get_obs(), reward, done, info

    def rad_to_target(self):
        """
        given the triangle:

                                direction
                                /       \
                               /         \
                            coxa LM-----coxa RM

        we want to match the angle between the 2 vectors: (direction,LF) and (direction,RF) and base.
        we need to:
        1. calculate the vectors
        2. calculate the angles r1,r2
        3. match the angle to match them: 0.5(r1-r2) thats how much we want to turn
        :return: angle of desired rotation
        """
        coxa_lf = self.get_pos('coxa_LM')
        coxa_rf = self.get_pos('coxa_RM')

        r1 = angle_between(self.direction[:2], coxa_rf[:2], coxa_lf[:2])
        r2 = angle_between(self.direction[:2], coxa_lf[:2], coxa_rf[:2])

        return 0.5 * (r2 - r1)  # minus the angle, i.e angle correction

    def do_simulation(self, ctrl, n_frames, callback=None, render=False):
        self.sim.data.ctrl[:] = ctrl
        for i in range(n_frames):
            if callback is not None:
                callback()
            self.sim.step()
            # self.ctrl_history.append(self.ctrl)
            if render and i % 100 == 0:
                # self.render(camera_name='side')
                frame = self.render(mode='rgb_array', camera_name='side')
                self.save_frame(frame)

    def save_frame(self, frame):
        if not os.path.isdir(f'{self.dirname}/frames'):
            pathlib.Path(f'{self.dirname}/frames').mkdir(parents=True, exist_ok=True)
        plt.imsave(f'{self.dirname}/frames/{time.time()}.png', frame)

    def get_obs(self):
        """
        return environment's observation
        """
        return self.qpos

    @property
    def map_joint_qpos(self):
        """
        map joint name to qpos position
        """
        return {joint.value: self.model.get_joint_qpos_addr(joint.value) for joint in JointNames}

    @property
    def map_joint_qvel(self):
        """
        map joint name to qvel position
        """
        return {joint.value: self.model.get_joint_qvel_addr(joint.value) for joint in JointNames}

    @property
    def qvel(self):
        return self.sim.data.qvel.copy()

    @property
    def qpos(self):
        return self.sim.data.qpos.copy()

    @property
    def ctrl(self):
        return self.sim.data.ctrl.copy()

    def index_of_body(self, name):
        return self.body_names.index(name)

    def get_pos(self, name):
        return self.sim.data.body_xpos[self.index_of_body(name)].copy()
    #
    # def save_ctrl(self):
    #     if len(self.ctrl_history) == 0:
    #         return
    #     plt.figure(figsize=(10, 10))
    #
    #     history = np.array(self.ctrl_history)  # i rows, len(ctrl) columns
    #
    #     for i, joint in enumerate(self.acuator_names):
    #         plt.plot(np.rad2deg(history.T[i]), label=joint)
    #
    #     plt.xlabel('step')
    #     plt.ylabel('degrees')
    #     plt.title('footfall')
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig(f'out/{time.time()}.png')
    #     self.ctrl_history.clear()
