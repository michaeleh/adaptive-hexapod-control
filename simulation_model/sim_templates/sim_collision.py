from kinematics.ik_algorithm import angles_to_target
from simulation_model.sim_templates.abstract_sim_template import AbstractSimTemplate


class SimCollision(AbstractSimTemplate):
    def __init__(self, map_joint_qpos):
        super().__init__()
        self.map_joint_qpos = map_joint_qpos
        self.contacts = []

    def eval(self, action, diff_pos):
        """
        clipping leg position due to collision
        :param diff_pos: leg -> relative change in pos

        """
        for leg, diff in diff_pos.items():
            coxa = self.map_joint_qpos[leg.coxa.value]
            femur = self.map_joint_qpos[leg.femur.value]
            tibia = self.map_joint_qpos[leg.tibia.value]
            joint_pos = [coxa, femur, tibia]
            action[joint_pos], e = angles_to_target(q=action[joint_pos], target=leg.rotate(diff))
        return action
