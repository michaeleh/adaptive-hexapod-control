import numpy as np
from gym.envs.mujoco import MujocoEnv
from scipy.spatial.transform import Rotation

from simulation_model.joint_types import JointNames, EENames
from simulation_model.sim_templates.sim_balance import SimBalance
from simulation_model.sim_templates.sim_collision import SimCollision
from simulation_model.sim_templates.sim_movement import SimMovement
from simulation_model.sim_templates.sim_balance import SimBalance

np.set_printoptions(suppress=True)


class HexapodEnv(MujocoEnv):

    def __init__(self, model_path, frame_skip):
        super().__init__(model_path, frame_skip)
        # initialize simulation templates
        self.movement_sim = SimMovement(self.qvel.shape,
                                        self.map_joint_qvel,
                                        self.map_joint_qpos,
                                        self.dt)
        self.collision_sim = SimCollision()
        self.balance_sim = SimBalance(self.dt)

    def reset_model(self):
        """
        reset simulation_model to zero state
        """
        self.set_state(np.zeros_like(self.sim.data.qpos), np.zeros_like(self.sim.data.qvel))

        self.movement_sim.reset()
        self.collision_sim.reset()
        self.balance_sim.reset()

        self.body_names = list(self.model.body_names)
        self.initial_ee_pos = np.array([self.sim.data.body_xpos[self.index_of_body(b.value)] for b in EENames])
        return self.get_obs()

    def step(self, action, render=False):
        if action.shape == (self.model.nq,):  # if step is action (or other mujoco's inner step runs)
            # Motion: pos after vanilla action
            qvel = self.movement_sim.eval(self.qpos, action)
            self.run_mujoco_sim(action, qvel, render=True)
            # Balance calculate orientation due to instability
            contact_pos = [self.sim.data.contact[i].pos for i in
                           range(self.sim.data.ncon)]  # find bodies in contact with floor or obstacle and get position

            action = self.balance_sim.eval(self.qpos, contact_pos)
            self.run_mujoco_sim(action, qvel, render=True)
            # sync legs in contact pos to stay at the same pos

        reward = 0
        done = False
        info = dict()
        return self.get_obs(), reward, done, info

    def run_mujoco_sim(self, action, qvel, render):
        """
        forward kinematics and mujoco sim
        """
        self.set_state(action, qvel)
        # apply physics simulation steps
        for _ in range(self.frame_skip):
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

    def index_of_body(self, name):
        return self.body_names.index(name)

    def axis_change(self):
        """
        body moves relative to the direction of all end effectors i.e the sync move
        :return: [x,y,z] change of end effectors
        """
        torso_current = self.get_body_pos('body:torso')
        ee_pos = np.array([self.get_body_pos(b.value) for b in EENames])
        # remove ee pos dependency in space by normalizing to body
        relative_pos = ee_pos - torso_current
        # change w.r.t initial pos
        diff = relative_pos - self.initial_ee_pos
        return diff.sum(axis=0)

    def get_body_pos(self, body_name):
        return self.sim.data.body_xpos[self.index_of_body(body_name)]

    def get_joint_pos(self, joint_name):
        return self.get_body_pos(joint_name.replace('joint', 'body'))

    @property
    def curr_rot(self):
        w, x, y, z = self.qpos[3:7]  # current orientation
        q = Rotation.from_quat([x, y, z, w])
        return q.as_euler('xyz', degrees=False)
