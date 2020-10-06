class DeltaLengths:
    # lengths in mm measured infusion360 using the provided stl files
    TIBIA_END_X = -10
    TIBIA_END_Y = 0
    TIBIA_END_Z = -97.29

    FEMUR_TIBIA_X = -52.77667098
    FEMUR_TIBIA_Y = 1.18366597
    FEMUR_TIBIA_Z = 43.06314346

    COXA_FEMUR_X = -63.548631
    COXA_FEMUR_Y = 0.1611743
    COXA_FEMUR_Z = -4.80630595


class JointIdx:
    COXA, FEMUR, TIBIA = range(3)
