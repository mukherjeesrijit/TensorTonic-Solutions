import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    matrix = np.asarray(matrix, dtype=float)
    if (axis and axis > matrix.ndim - 1) or (matrix.ndim > 2) or (norm_type == 'invalid'):
        return None
    norm_map = {'l1': 1, 'l2': (None if axis is None else 2), 'max': np.inf}
    ord_val = norm_map[norm_type]
    norm = np.linalg.norm(matrix, ord=ord_val, axis=axis, keepdims=True)
    result = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
    return result