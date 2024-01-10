"""
Distances between parts of the hexapod
"""


class DeltaLengths:
    TIBIA_END_X = 0.0178143
    TIBIA_END_Y = 0.02805175
    TIBIA_END_Z = 0.12521492

    FEMUR_TIBIA_X = -0.03214914
    FEMUR_TIBIA_Y = 0.06157166
    FEMUR_TIBIA_Z = 0.01311938

    COXA_FEMUR_X = 0.01486056
    COXA_FEMUR_Y = 0.04842855
    COXA_FEMUR_Z = 0.02454252


class JointIdx:
    COXA, FEMUR, TIBIA = range(3)
