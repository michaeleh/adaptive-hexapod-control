import numpy as np


def angle_between(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle


def rotate_vec(vec, deg):
    rad = np.deg2rad(deg)
    '''
    x' = x cos θ − y sin θ
    y' = x sin θ + y cos θ
    '''
    x = vec[0]
    y = vec[1]
    z = vec[2]
    return np.array([
        x * np.cos(rad) - y * np.sin(rad),
        x * np.sin(rad) + y * np.cos(rad),
        z
    ])
