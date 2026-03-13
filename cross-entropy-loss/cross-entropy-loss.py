import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """

    y_true, y_pred = map(np.array, [y_true, y_pred])
    y_pred = np.clip(y_pred, 1e-15, 1.0)
    y_pred = -np.log(y_pred)
    n,c = y_pred.shape
    one_hots = np.zeros((n,c))
    one_hots[np.arange(n),y_true] = 1 
    loss_list = np.einsum('ij,ij -> i',y_pred,one_hots)
    return np.mean(loss_list)