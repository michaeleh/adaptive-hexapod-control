from enum import Enum

import numpy as np

from kinematics.joint_kinematics import HexapodLegKinematic, KinematicNumericImpl


class Optimizer(Enum):
    """ Designation of an optimization method for inverse kinematic

    We support two optimization methods for inverse kinematic:
    1. Standard resolved motion (STD): Based on Pseudo-inversed jacobian
    2. Dampened least squares method (DLS) or the Levenberg–Marquardt algorithm:
        see https://en.wikipedia.org/wiki/Levenberg–Marquardt_algorithm for a detailed description
    """

    STD = 1
    DLS = 2


def angles_to_target(q, target, model: HexapodLegKinematic = KinematicNumericImpl(), max_iter=10000, error_thold=0.01,
                     kp=0.1, optimizer=Optimizer.STD):
    """
    Giving arm object, a target and optimizer, provides the required set of control signals

    Returns the optimizing trajectory, error trace and arm configuration to achieve the target.
    Target is defined in relative to the EE null position

    :param q: angles of joints 3x1 np array
    :param model: Kinematic environment of a leg. inherits from HexapodLegKinematic
    :param target: target pos to get to
    :param max_iter: max iter to perform.
    :param error_thold: continue until reached error threshold
    :param kp: Proportional gain term
    :param optimizer: Optimizer from enum.
    :return: 3 joint angles to achieve target.
    """
    q = q.copy()
    pos_ee = model.calc_xyz(q)  # Current operational position of the arm
    pos_target = pos_ee + target  # Target operational position
    for step in range(max_iter):

        pos_ee = model.calc_xyz(q)  # Get current EE position
        pos_diff = pos_target - pos_ee  # Get vector to target
        error = np.linalg.norm(pos_diff)  # Distance to target
        # Stop when within 1mm accurancy (arm mechanical accurancy limit)
        if error < error_thold:
            break
        ux = pos_diff * kp  # direction of movement
        J_x = model.calc_J(q)  # Calculate the jacobian

        # Solve inverse kinematics according to the designated optimizer
        if optimizer is Optimizer.STD:  # Standard resolved motion
            u = np.dot(np.linalg.pinv(J_x), ux)

        elif optimizer is Optimizer.DLS:  # Dampened least squares method
            u = np.dot(J_x.T, np.linalg.solve(np.dot(J_x, J_x.T) + np.eye(3) * 0.001, ux))

        q += u

    if error > 1:
        print('error', error)
    return q, error
