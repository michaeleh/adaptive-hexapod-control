import numpy as np
from gym.envs.mujoco import MujocoEnv

from joint_types import JointNames

np.set_printoptions(suppress=True)


class HexapodEnv(MujocoEnv):

    def __init__(self, model_path, frame_skip):
        super().__init__(model_path, frame_skip)

    def reset_model(self):
        self.set_state(np.zeros_like(self.sim.data.qpos), np.zeros_like(self.sim.data.qvel))
        return self.get_obs()

    def step(self, action):
        if action.shape == (self.model.nq,):
            qvel = np.zeros_like(self.sim.data.qvel.copy())

            diff_pos = action - self.get_obs()
            qvel_id = self.map_joint_qvel()
            for joint, idx in self.map_joint_qpos().items():
                qvel[qvel_id[joint]] = diff_pos[idx]
            qvel[1] = np.linalg.norm(qvel)
            self.set_state(action, qvel)
            for _ in range(self.frame_skip):
                self.sim.step()
                self.render()

        reward = 0
        done = False
        info = dict()
        return self.get_obs(), reward, done, info

    def get_obs(self):
        return self.sim.data.qpos.copy()

    def map_joint_qpos(self):
        return {joint.value: self.model.get_joint_qpos_addr(joint.value) for joint in JointNames}

    def map_joint_qvel(self):
        return {joint.value: self.model.get_joint_qvel_addr(joint.value) for joint in JointNames}

    @property
    def qvel(self):
        return self.sim.data.qvel.copy()
