import time

import matplotlib.pyplot as plt
from typing import Dict

import numpy as np
from gym.envs.mujoco import MujocoEnv

from environment.defaults import fill_pos_defaults
from environment.joint_types import JointNames

np.set_printoptions(suppress=True)


class HexapodEnv(MujocoEnv):

    def __init__(self, model_path, frame_skip):
        super().__init__(model_path, frame_skip)
        self.ctrl_history = []

    def reset_model(self):
        """
        reset environment to zero state
        """
        self.acuator_names = list(self.model.actuator_names)
        # self.save_ctrl()
        qpos = self.qpos * 0
        # fill_pos_defaults(qpos, self.map_joint_qpos)
        self.set_state(qpos, 0 * self.qvel)
        self.body_names = list(self.model.body_names)
        self.do_simulation(0 * self.ctrl, self.frame_skip)
        return self.get_obs()

    def step(self, action: Dict[str, float], render=False):
        '''
        step simulation and implement action
        :param action: dict of joint_name -> desired angle
        :return: new observation
        '''
        if isinstance(action, Dict):  # if step is action (or other mujoco's inner step runs)
            # Motion: pos after vanilla action
            ctrl = self.ctrl   # dont change current angles

            # replace changes values
            for joint, angle in action.items():
                idx = self.acuator_names.index(joint)
                ctrl[idx] = angle
            self.do_simulation(ctrl, self.frame_skip, render)
        reward = 0
        done = False
        info = dict()
        return self.get_obs(), reward, done, info

    def do_simulation(self, ctrl, n_frames, render=True):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()
            self.ctrl_history.append(self.ctrl)
            if render:
                self.render()

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
        return self.sim.data.body_xpos[self.index_of_body(name)]

    def save_ctrl(self):
        if len(self.ctrl_history) == 0:
            return
        plt.figure(figsize=(10, 10))

        history = np.array(self.ctrl_history)  # i rows, len(ctrl) columns

        for i, joint in enumerate(self.acuator_names):
            plt.plot(np.rad2deg(history.T[i]), label=joint)

        plt.xlabel('step')
        plt.ylabel('degrees')
        plt.title('footfall')
        plt.legend()
        plt.grid()
        plt.savefig(f'out/{time.time()}.png')
        self.ctrl_history.clear()
