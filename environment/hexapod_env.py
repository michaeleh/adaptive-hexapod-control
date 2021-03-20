from typing import Dict

import numpy as np
from gym.envs.mujoco import MujocoEnv

from environment.joint_types import JointNames

np.set_printoptions(suppress=True)


class HexapodEnv(MujocoEnv):

    def __init__(self, model_path, frame_skip):
        super().__init__(model_path, frame_skip)

    def reset_model(self):
        """
        reset environment to zero state
        """
        self.set_state(0 * self.qpos, 0 * self.qvel)
        self.body_names = list(self.model.body_names)
        self.acuator_names = list(self.model.actuator_names)
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
            ctrl = self.ctrl  # dont change current angles
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
