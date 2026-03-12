import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).
    """

    points = np.array(points, dtype=float)
    points = np.atleast_2d(points).T
    R2z = [[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]
    points[:2] = R2z@points[:2]
    return np.squeeze(points).T