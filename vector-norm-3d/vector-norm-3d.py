import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """

    v = np.array(v, dtype = np.float64)
    v_2d = np.atleast_2d(v)
    norms = np.linalg.norm(v_2d, axis = 1)
    # norms[norms == 0] = 1
    # print(norms)
    return norms[0] if v.ndim == 1 else norms