import numpy as np
from gym.envs.mujoco import MujocoEnv
from scipy.spatial.transform import Rotation

from kinematics.ik_algorithm import angles_to_target
from simulation_model.joint_types import JointNames, EENames
from simulation_model.leg import leg_from_geom
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
        self.collision_sim = SimCollision(self.map_joint_qpos)
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

        self.Lx = np.linalg.norm(self.get_body_pos(EENames.EE_RM.value)[0] - self.get_body_pos(EENames.EE_LM.value)[0])
        self.Ly = np.linalg.norm(self.get_body_pos(EENames.EE_RR.value)[1] - self.get_body_pos(EENames.EE_RF.value)[1])
        return self.get_obs()

    def step(self, action, render=False):
        if action.shape == (self.model.nq,):  # if step is action (or other mujoco's inner step runs)
            # Motion: pos after vanilla action
            self.run_mujoco_sim(action, render=True)

            # find bodies in contact with floor or obstacle and get position
            # clip action due to collision
            contacts = self.get_legs_contacts()
            diff = {leg: pos - self.get_body_pos(leg.ee.value) for leg, pos in contacts.items()}
            action = self.collision_sim.eval(self.qpos, diff)
            self.run_mujoco_sim(action, render=True)
            # Balance calculate orientation due to instability
            action = self.balance_sim.eval(self.qpos, contacts.values(), self.Lx, self.Ly)
            self.run_mujoco_sim(action, render=False)

            # sync legs in contact pos to stay at the same pos
            # action = self.qpos
            # for leg, pos in contacts.items():
            #     ee = leg.ee
            #
            #     curr_pos = self.get_body_pos(ee.value)
            #     diff = pos - curr_pos
            #     coxa = self.map_joint_qpos[leg.coxa.value]
            #     femur = self.map_joint_qpos[leg.femur.value]
            #     tibia = self.map_joint_qpos[leg.tibia.value]
            #     joint_pos = [coxa, femur, tibia]
            #     # nice idea not working but not needed for now FIXME
            #     # action[joint_pos], e = angles_to_target(q=self.qpos[joint_pos], target=-leg.rotate(diff))
            #
            # self.run_mujoco_sim(action, render=True)
        self.render()
        reward = 0
        done = False
        info = dict()
        return self.get_obs(), reward, done, info

    def get_legs_contacts(self):
        """
        :return: each leg contact position
        """
        contacts = {}  # where each hexapod's geom had contact
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = self.sim.model.geom_id2name(contact.geom1)
            geom2 = self.sim.model.geom_id2name(contact.geom2)
            # only 1 leg thus either 1 or 2
            leg = leg_from_geom(geom1)
            if leg is None:
                leg = leg_from_geom(geom2)
            contacts[leg] = contact.pos
        return contacts

    def run_mujoco_sim(self, action, render):
        """
        forward kinematics and mujoco sim
        """
        qvel = self.movement_sim.eval(self.qpos, action)

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

    # def axis_change(self):
    #     """
    #     body moves relative to the direction of all end effectors i.e the sync move
    #     :return: [x,y,z] change of end effectors
    #     """
    #     torso_current = self.get_body_pos('body:torso')
    #     ee_pos = np.array([self.get_body_pos(b.value) for b in EENames])
    #     # remove ee pos dependency in space by normalizing to body
    #     relative_pos = ee_pos - torso_current
    #     # change w.r.t initial pos
    #     diff = relative_pos - self.initial_ee_pos
    #     return diff.sum(axis=0)

    def get_body_pos(self, body_name):
        return self.sim.data.body_xpos[self.index_of_body(body_name)]

    def get_joint_pos(self, joint_name):
        return self.get_body_pos(joint_name.replace('joint', 'body'))

    @property
    def curr_rot(self):
        w, x, y, z = self.qpos[3:7]  # current orientation
        q = Rotation.from_quat([x, y, z, w])
        return q.as_euler('xyz', degrees=False)
