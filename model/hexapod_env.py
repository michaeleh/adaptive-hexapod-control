import numpy as np
from gym.envs.mujoco import MujocoEnv

from model.joint_types import JointNames

np.set_printoptions(suppress=True)


class HexapodEnv(MujocoEnv):

    def __init__(self, model_path, frame_skip):
        super().__init__(model_path, frame_skip)

    def reset_model(self):
        """
        reset model to zero state
        """
        self.set_state(np.zeros_like(self.sim.data.qpos), np.zeros_like(self.sim.data.qvel))
        return self.get_obs()

    def step(self, action):
        if action.shape == (self.model.nq,):

            qvel = np.zeros_like(self.qvel)  # copy empty velocity shape
            diff_pos = action - self.get_obs()  # difference in position is velocity
            qvel_id = self.map_joint_qvel()
            for joint, idx in self.map_joint_qpos().items():
                qvel[qvel_id[joint]] = diff_pos[idx] / self.dt  # apply velocity for each joint

            '''
            qpos[0]~qpos[6] corresponds to the 'root' joint cartesian position (qpos[0]~qpos[2]) and orientation
             (qpos[3]~qpos[6]), and qvel[0]~qvel[5] correspond to the 'root' joint velocity,
            translational (qvel[0]~qvel[2]) and rotational (qvel[3]~qvel[5]).
            Note that for orientation you follow the quaternion notation thus need four element
            but for velocity you use angular velocity which consists of 3 elements.
            '''
            qvel[0:3] = diff_pos[0:3]  # set xyz velocity
            self.set_state(action, qvel)
            # apply physics simulation steps
            for _ in range(self.frame_skip):
                self.sim.step()
                self.render()

        reward = 0
        done = False
        info = dict()
        return self.get_obs(), reward, done, info

    def get_obs(self):
        """
        return environment's observation
        """
        return self.qpos

    def map_joint_qpos(self):
        """
        map joint name to qpos position
        """
        return {joint.value: self.model.get_joint_qpos_addr(joint.value) for joint in JointNames}

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
