import numpy as np


def xy2polar(x, y):
    r = np.linalg.norm([x, y])
    theta = np.arctan2(y, x)
    return r, theta


def polar2xy(r, rad):
    x = r * np.cos(rad)
    y = r * np.sin(rad)
    return x, y


def cartesian_change(r, src, dst=0):
    x0, y0 = polar2xy(r, src)
    x1, y1 = polar2xy(r, dst)
    return [x1 - x0, y1 - y0]  # y,z plane
