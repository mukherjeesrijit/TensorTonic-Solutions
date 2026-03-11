import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_pred, y_true = map(np.array, [y_pred, y_true])
    return np.linalg.norm(y_pred-y_true)**2/len(y_pred)
