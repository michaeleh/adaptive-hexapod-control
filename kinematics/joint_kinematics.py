from abc import ABC, abstractmethod
from pprint import pprint
from typing import List

import numpy as np
import sympy as sp
from numpy import sin, cos

from kinematics.constants import DeltaLengths, JointIdx


class HexapodLegKinematic(ABC):
    n_joints = 3

    @abstractmethod
    def calc_xyz(self, q):
        pass

    def calc_J(self, q):
        pass


class KinematicSymbolicImpl(HexapodLegKinematic):

    def __init__(self) -> None:
        super().__init__()
        self.q_body_coxa = sp.Symbol('q_body_coxa')
        self.q_coxa_femur = sp.Symbol('q_coxa_femur')
        self.q_femur_tibia = sp.Symbol('q_femur_tibia')
        self.Tx = self._build_Tx()
        self.J = self._build_J()

    def _build_Tx(self):
        """
        calculating Tx for kinematics using translation and rotation matrix
        """
        q_body_coxa = self.q_body_coxa
        q_coxa_femur = self.q_coxa_femur
        q_femur_tibia = self.q_femur_tibia
        # rotation around z axis
        T_body_coxa = sp.Matrix([
            [sp.cos(q_body_coxa), -sp.sin(q_body_coxa), 0, 0],
            [sp.sin(q_body_coxa), sp.cos(q_body_coxa), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        # rotation around y axis
        T_coxa_femur = sp.Matrix([
            [sp.cos(q_coxa_femur), 0, sp.sin(q_coxa_femur), DeltaLengths.COXA_FEMUR_X],
            [0, 1, 0, DeltaLengths.COXA_FEMUR_Y],
            [-sp.sin(q_coxa_femur), 0, sp.cos(q_coxa_femur), DeltaLengths.COXA_FEMUR_Z],
            [0, 0, 0, 1]
        ])

        # rotation around y axis
        T_femur_tibia = sp.Matrix([
            [sp.cos(q_femur_tibia), 0, sp.sin(q_femur_tibia), DeltaLengths.FEMUR_TIBIA_X],
            [0, 1, 0, DeltaLengths.FEMUR_TIBIA_Y],
            [-sp.sin(q_femur_tibia), 0, sp.cos(q_femur_tibia), DeltaLengths.FEMUR_TIBIA_Z],
            [0, 0, 0, 1]
        ])

        # no rotation
        T_tibia_ee = sp.Matrix([
            [1, 0, 0, DeltaLengths.TIBIA_END_X],
            [0, 1, 0, DeltaLengths.TIBIA_END_Y],
            [0, 0, 1, DeltaLengths.TIBIA_END_Z],
            [0, 0, 0, 1],
        ])
        T = T_body_coxa * T_coxa_femur * T_femur_tibia * T_tibia_ee
        x = sp.Matrix([0, 0, 0, 1])  # starting point is always at 0,0,0. only qs changes
        return T * x

    @staticmethod
    def _evaluate_symbols(matrix, q: List[float]):
        """
        substitute angles and return result
        :param matrix: to perform the operation on
        :param q: q0 = boxy coxa angle, q1 = coxa femur angle, q2 = femur tibia angle
        :return: x,y,z of position
        """
        return np.array(matrix.subs([('q_body_coxa', q[0]),
                                     ('q_coxa_femur', q[1]),
                                     ('q_femur_tibia', q[2])]), dtype=np.float)

    def calc_xyz(self, q: List[float]):
        """
        list of angles to calculate foot-tip xyz
        :param q: q0 = boxy coxa angle, q1 = coxa femur angle, q2 = femur tibia angle
        :return: final foot-tip position xyz
        """
        return self._evaluate_symbols(self.Tx, q).flatten()[:-1]  # array of only x,y,z

    def calc_J(self, q: List[float]):
        return self._evaluate_symbols(self.J, q)

    def _build_J(self):
        """
        :return: jacobian of transformation matrix
        """
        q = [self.q_body_coxa, self.q_coxa_femur, self.q_femur_tibia]
        n_axis = 3
        J = sp.Matrix.ones(n_axis, len(q))
        for i in range(n_axis):  # x, y, z
            for j in range(len(q)):  # Four joints
                partial_diff = self.Tx[i].diff(q[j])
                J[i, j] = sp.simplify(partial_diff)
        return J


class KinematicNumericImpl(HexapodLegKinematic):
    def calc_xyz(self, q):
        """
        :param q: In radians
        :return:
        """
        q_body_coxa = q[JointIdx.COXA]
        q_coxa_femur = q[JointIdx.FEMUR]
        q_femur_tibia = q[JointIdx.TIBIA]

        return np.array([
            [-12.468 * sin(q_body_coxa) + 11.671 * sin(q_coxa_femur) * sin(q_femur_tibia) * cos(
                q_body_coxa) - 137.168 * sin(q_coxa_femur) * cos(q_body_coxa) * cos(q_femur_tibia) - 5.712 * sin(
                q_coxa_femur) * cos(q_body_coxa) - 137.168 * sin(q_femur_tibia) * cos(q_body_coxa) * cos(
                q_coxa_femur) - 11.671 * cos(q_body_coxa) * cos(q_coxa_femur) * cos(q_femur_tibia) + 67.744 * cos(
                q_body_coxa) * cos(q_coxa_femur) + 62.451 * cos(q_body_coxa)],
            [11.671 * sin(q_body_coxa) * sin(q_coxa_femur) * sin(q_femur_tibia) - 137.168 * sin(q_body_coxa) * sin(
                q_coxa_femur) * cos(q_femur_tibia) - 5.712 * sin(q_body_coxa) * sin(q_coxa_femur) - 137.168 * sin(
                q_body_coxa) * sin(q_femur_tibia) * cos(q_coxa_femur) - 11.671 * sin(q_body_coxa) * cos(
                q_coxa_femur) * cos(q_femur_tibia) + 67.744 * sin(q_body_coxa) * cos(q_coxa_femur) + 62.451 * sin(
                q_body_coxa) + 12.468 * cos(q_body_coxa)],
            [137.168 * sin(q_coxa_femur) * sin(q_femur_tibia) + 11.671 * sin(q_coxa_femur) * cos(
                q_femur_tibia) - 67.744 * sin(q_coxa_femur) + 11.671 * sin(q_femur_tibia) * cos(
                q_coxa_femur) - 137.168 * cos(q_coxa_femur) * cos(q_femur_tibia) - 5.712 * cos(
                q_coxa_femur) + 7.105]]).flatten()

    def calc_J(self, q):
        """
        :param q: In radians
        :return:
        """
        q_body_coxa = q[JointIdx.COXA]
        q_coxa_femur = q[JointIdx.FEMUR]
        q_femur_tibia = q[JointIdx.TIBIA]

        return np.array([
            [5.712 * sin(q_body_coxa) * sin(q_coxa_femur) + 137.168 * sin(q_body_coxa) * sin(
                q_coxa_femur + q_femur_tibia) - 67.744 * sin(q_body_coxa) * cos(q_coxa_femur) + 11.671 * sin(
                q_body_coxa) * cos(q_coxa_femur + q_femur_tibia) - 62.451 * sin(q_body_coxa) - 12.468 * cos(
                q_body_coxa), (-67.744 * sin(q_coxa_femur) + 11.671 * sin(q_coxa_femur + q_femur_tibia) - 5.712 *
                               cos(q_coxa_femur) - 137.168 * cos(q_coxa_femur + q_femur_tibia)) * cos(q_body_coxa),
             (11.671 * sin(q_coxa_femur + q_femur_tibia) - 137.168 * cos(q_coxa_femur + q_femur_tibia)) * cos(
                 q_body_coxa)],
            [-12.468 * sin(q_body_coxa) - 5.712 * sin(q_coxa_femur) * cos(q_body_coxa) - 137.168 * sin(
                q_coxa_femur + q_femur_tibia) * cos(q_body_coxa) + 67.744 * cos(q_body_coxa) * cos(
                q_coxa_femur) - 11.671 * cos(q_body_coxa) * cos(q_coxa_femur + q_femur_tibia) + 62.451 * cos(
                q_body_coxa), (-67.744 * sin(q_coxa_femur) + 11.671 * sin(q_coxa_femur + q_femur_tibia) - 5.712 *
                               cos(q_coxa_femur) - 137.168 * cos(q_coxa_femur + q_femur_tibia)) * sin(q_body_coxa),
             (11.671 * sin(q_coxa_femur + q_femur_tibia) - 137.168 * cos(q_coxa_femur + q_femur_tibia)) * sin(
                 q_body_coxa)],
            [0, 5.712 * sin(q_coxa_femur) + 137.168 * sin(q_coxa_femur + q_femur_tibia) - 67.744 * cos(
                q_coxa_femur) + 11.671 * cos(q_coxa_femur + q_femur_tibia),
             137.168 * sin(q_coxa_femur + q_femur_tibia) + 11.671 * cos(q_coxa_femur + q_femur_tibia)]])


if __name__ == '__main__':
    def print_symbolic():
        k = KinematicSymbolicImpl()
        pprint(k.Tx)
        print()
        pprint(k.J)


    def assert_impls():
        joints = np.deg2rad(np.array([10, 20, 30]))
        k1 = KinematicSymbolicImpl()
        k2 = KinematicNumericImpl()

        xyz = k1.calc_xyz(joints)
        xyz2 = k2.calc_xyz(joints)
        j = k1.calc_J(joints)
        j2 = k2.calc_J(joints)
        assert np.array_equal(np.round(xyz, 3), np.round(xyz2, 3))
        assert np.array_equal(np.round(j, 3), np.round(j2, 3))


    assert_impls()
