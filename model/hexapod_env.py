import numpy as np
from gym.envs.mujoco import MujocoEnv

from kinematics.quaternion import transform, inverse, rotation_vec
from model.joint_types import JointNames
from model.leg import forward_vec

np.set_printoptions(suppress=True)


class HexapodEnv(MujocoEnv):

    def __init__(self, model_path, frame_skip):
        super().__init__(model_path, frame_skip)

    def reset_model(self):
        self.set_state(np.zeros_like(self.sim.data.qpos), np.zeros_like(self.sim.data.qvel))
        return self.get_obs()

    def step(self, action):
        if action.shape == (self.model.nq,):
            qvel = np.zeros_like(self.qvel)
            diff_pos = action - self.get_obs()
            qvel_id = self.map_joint_qvel()
            for joint, idx in self.map_joint_qpos().items():
                qvel[qvel_id[joint]] = diff_pos[idx] / self.dt

            '''
            qpos[0]~qpos[6] corresponds to the 'root' joint cartesian position (qpos[0]~qpos[2]) and orientation (qpos[3]~qpos[6]),
            and qvel[0]~qvel[5] correspond to the 'root' joint velocity,
            translational (qvel[0]~qvel[2]) and rotational (qvel[3]~qvel[5]).
            Note that for orientation you follow the quaternion notation thus need four element
            but for velocity you use angular velocity which consists of 3 elements.
            '''

            qvel[0:3] = diff_pos[0:3]
            self.set_state(action, qvel)
            for _ in range(self.frame_skip):
                self.sim.step()
                self.render()

        reward = 0
        done = False
        info = dict()
        return self.get_obs(), reward, done, info

    def get_obs(self):
        return self.qpos

    def map_joint_qpos(self):
        return {joint.value: self.model.get_joint_qpos_addr(joint.value) for joint in JointNames}

    def map_joint_qvel(self):
        return {joint.value: self.model.get_joint_qvel_addr(joint.value) for joint in JointNames}

    @property
    def qvel(self):
        return self.sim.data.qvel.copy()

    @property
    def qpos(self):
        return self.sim.data.qpos.copy()
