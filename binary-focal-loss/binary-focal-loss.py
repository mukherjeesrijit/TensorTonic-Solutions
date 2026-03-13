import numpy as np 

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """

    predictions, targets = map(np.array, [predictions, targets])
    p = predictions*targets + (1-predictions)*(1-targets)
    L = -alpha* (1-p)**gamma * np.log(p)
    return np.mean(L)