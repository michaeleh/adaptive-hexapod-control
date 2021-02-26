from scipy.spatial.transform import Rotation

from simulation_model.sim_templates.abstract_sim_template import AbstractSimTemplate
import numpy as np


class SimMovement(AbstractSimTemplate):
    def __init__(self, qvel_shape, qvel_id, qpos_id, dt):
        super().__init__()
        self.dt = dt  # sim dt
        self.qvel_id = qvel_id  # qvel id dict (name-> qvel place)
        self.qpos_id = qpos_id
        self.zero_qvel = np.zeros(qvel_shape)

    def eval(self, curr_pos, new_pos) -> np.array:
        """
        calculates qvel from pos. returns velocity as qvel.
        """
        qvel = self.zero_qvel.copy()  # copy empty velocity shape
        diff_pos = new_pos - curr_pos  # difference in position is velocity
        for joint, idx in self.qpos_id.items():
            joint_idx = self.qvel_id[joint]
            qvel[joint_idx] = diff_pos[idx] * self.dt  # apply velocity for each joint

        '''
        qpos[0]~qpos[6] corresponds to the 'root' joint cartesian position (qpos[0]~qpos[2]) and orientation
         (qpos[3]~qpos[6]), and qvel[0]~qvel[5] correspond to the 'root' joint velocity,
        translational (qvel[0]~qvel[2]) and rotational (qvel[3]~qvel[5]).
        Note that for orientation you follow the quaternion notation thus need four element
        but for velocity you use angular velocity which consists of 3 elements.
        '''
        qvel[0:3] = diff_pos[0:3]  # set xyz velocity
        w1, x1, y1, z1 = new_pos[3:7]
        w2, x2, y2, z2 = curr_pos[3:7]
        # q1 = Rotation.from_quat([x1, y1, z1, w1])
        # q2 = Rotation.from_quat([x2, y2, z2, w2])
        # qvel[3:6] = q1.as_euler('xyz', degrees=False) - q2.as_euler('xyz', degrees=False)
        return qvel
