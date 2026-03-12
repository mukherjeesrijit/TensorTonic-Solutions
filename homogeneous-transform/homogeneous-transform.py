import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """

    T, points = map(np.array, [T, points])
    points = np.atleast_2d(points)
    points = np.insert(points, points.shape[1], 1, axis = 1)
    points = points.T
    results = (T@points)[:3].T
    return np.squeeze(results)