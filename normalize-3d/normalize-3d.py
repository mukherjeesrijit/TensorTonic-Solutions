import numpy as np

def normalize_3d(v):
    v = np.array(v, dtype=np.float64)
    v_2d = np.atleast_2d(v)
    norms = np.linalg.norm(v_2d, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    result = v_2d / norms
    return result.reshape(v.shape)
