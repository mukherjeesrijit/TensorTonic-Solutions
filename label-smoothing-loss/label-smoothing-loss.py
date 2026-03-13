import numpy as np 

def label_smoothing_loss(predictions, target, epsilon):
    
    """
    Compute cross-entropy loss with label smoothing.
    """

    p = np.array(predictions)
    k = len(p)
    hots = np.arange(k)
    q = np.where(hots == target, (1-epsilon) + epsilon/k ,epsilon/k)
    L = -np.dot(q,np.log(p))
    return L