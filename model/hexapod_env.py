import numpy as np
from gym.envs.mujoco import MujocoEnv
from scipy.spatial.transform import Rotation

from model.joint_types import JointNames, EENames

np.set_printoptions(suppress=True)


class HexapodEnv(MujocoEnv):

    def __init__(self, model_path, frame_skip):
        super().__init__(model_path, frame_skip)

    def reset_model(self):
        """
        reset model to zero state
        """
        self.set_state(np.zeros_like(self.sim.data.qpos), np.zeros_like(self.sim.data.qvel))
        self.body_names = list(self.model.body_names)
        self.initial_ee_pos = np.array([self.sim.data.body_xpos[self.index_of_body(b.value)] for b in EENames])
        return self.get_obs()

    def step(self, action, render=False):
        if action.shape == (self.model.nq,):

            qvel = np.zeros_like(self.qvel)  # copy empty velocity shape
            diff_pos = action - self.get_obs()  # difference in position is velocity
            qvel_id = self.map_joint_qvel()
            for joint, idx in self.map_joint_qpos().items():
                qvel[qvel_id[joint]] = diff_pos[idx] * self.dt  # apply velocity for each joint

            '''
            qpos[0]~qpos[6] corresponds to the 'root' joint cartesian position (qpos[0]~qpos[2]) and orientation
             (qpos[3]~qpos[6]), and qvel[0]~qvel[5] correspond to the 'root' joint velocity,
            translational (qvel[0]~qvel[2]) and rotational (qvel[3]~qvel[5]).
            Note that for orientation you follow the quaternion notation thus need four element
            but for velocity you use angular velocity which consists of 3 elements.
            '''
            qvel[0:3] = diff_pos[0:3]  # set xyz velocity
            w1, x1, y1, z1 = action[3:7]
            w2, x2, y2, z2 = self.get_obs()[3:7]
            q1 = Rotation.from_quat([x1, y1, z1, w1])
            q2 = Rotation.from_quat([x2, y2, z2, w2])
            qvel[3:6] = q1.as_euler('xyz', degrees=False) - q2.as_euler('xyz', degrees=False)

            self.set_state(action, qvel)
            # apply physics simulation steps
            for _ in range(self.frame_skip):
                self.sim.step()
                if render:
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

    def index_of_body(self, name):
        return self.body_names.index(name)

    def axis_change(self):
        torso_current = self.sim.data.body_xpos[self.index_of_body('body:torso')]
        ee_pos = np.array([self.sim.data.body_xpos[self.index_of_body(b.value)] for b in EENames])

        relative_pos = ee_pos - torso_current
        diff = relative_pos - self.initial_ee_pos

        return diff.sum(axis=0)

    def hexa_h(self):
        return max(self.data.body_xpos[self.index_of_body(JointNames.COXA_RM.value.replace('joint', 'body'))][2],
                   self.data.body_xpos[self.index_of_body(JointNames.COXA_LM.value.replace('joint', 'body'))][2])

    def get_pos(self, joint_name):
        return np.array(self.data.body_xpos[self.index_of_body(joint_name.replace('joint', 'body'))])
