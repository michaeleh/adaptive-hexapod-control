import numpy as np


def transform(x, y, z, rad):
    c = np.cos(rad / 2)
    s = np.sin(rad / 2)
    return np.array([c, x * s, y * s, z * s])


def inverse(vec):
    rad = np.arccos(vec[0]) * 2
    s = np.sin(rad / 2)
    if s == 0:
        return np.zeros(4)
    x = vec[1] / s
    y = vec[2] / s
    z = vec[3] / s
    return x, y, z, rad


def rotation_vec(vec):
    rotationx, rotationy, rotationz, rad = inverse(vec)
    return np.array([rotationx, rotationy, rotationz]) * rad
