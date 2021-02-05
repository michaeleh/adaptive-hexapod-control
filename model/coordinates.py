import numpy as np


def cartesian_change(r, src, dst):
    x0 = r * np.cos(src)
    y0 = r * np.sin(src)
    x1 = r * np.cos(dst)
    y1 = r * np.sin(dst)
    return x1 - x0, y1 - y0