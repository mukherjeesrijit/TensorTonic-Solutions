import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    y_true, y_score = map(np.array, [y_true, y_score])

    if reduction == "mean":
        return np.mean(np.maximum(0,margin-y_true*y_score))
    else:
        return np.sum(np.maximum(0,margin-y_true*y_score))