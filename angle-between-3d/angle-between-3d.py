import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """

    v = np.array(v)
    w = np.array(w)
    costheta = np.dot(v,w)/(np.linalg.norm(v)*np.linalg.norm(w))
    return np.arccos(costheta)