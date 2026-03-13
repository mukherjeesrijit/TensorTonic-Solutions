import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """

    p,y = map(np.array, [p,y])
    term1 = -(1-p)**gamma * y * np.log(p)
    term2 =  -(p)**gamma * (1-y) * np.log(1-p)
    result = term1 + term2
    return np.mean(result)