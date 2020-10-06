import numpy as np
from gym.envs.mujoco import MujocoEnv

from joint_types import JointNames

np.set_printoptions(suppress=True)


class HexapodEnv(MujocoEnv):

    def __init__(self, model_path, frame_skip):
        super().__init__(model_path, frame_skip)

    def reset_model(self):
        return self._get_obs()

    def step(self, action):
        qpos = action
        qvel = self.sim.data.qvel

        if qpos.shape == (self.model.nq,):
            self.set_state(qpos, qvel)
            self.sim.step()

        observation = self._get_obs()
        reward = 0
        done = False
        info = dict()
        return observation, reward, done, info

    def _get_obs(self):
        return self.sim.data.qpos.copy()

    def map_joint_pos(self):
        return {joint.value: self.model.get_joint_qpos_addr(joint.value) for joint in JointNames}
