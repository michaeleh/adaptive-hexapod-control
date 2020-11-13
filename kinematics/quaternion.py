import numpy as np

"""
Body orientation uses quaternion, in order to calculate velocity or apply orientation
one needs to convert back and fourth.
"""


def transform(x, y, z, rad):
    """
    given x,y,z unit axis vector and rotation angle, convert to quaternion notation.
    """
    c = np.cos(rad / 2)
    s = np.sin(rad / 2)
    return np.array([c, x * s, y * s, z * s])


def inverse(vec):
    """
    Inverse quaternion notation.
    """
    rad = np.arccos(vec[0]) * 2
    s = np.sin(rad / 2)
    if s == 0:
        return np.zeros(4)
    x = vec[1] / s
    y = vec[2] / s
    z = vec[3] / s
    return x, y, z, rad


def rotation_vec(vec):
    """
    convert quaternion notation to rotation vector per axis.
    """
    rotationx, rotationy, rotationz, rad = inverse(vec)
    return np.array([rotationx, rotationy, rotationz]) * rad
