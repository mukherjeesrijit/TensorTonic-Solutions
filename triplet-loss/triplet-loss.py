import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """

    anchor, positive, negative = map(np.atleast_2d, [anchor, positive, negative])
    dist = lambda x: np.linalg.norm(x, axis=1)**2
    pos = dist(anchor-positive)
    neg = dist(anchor-negative)
    losses = np.maximum(0,pos-neg+margin)   
    loss = np.mean(losses)
    return loss
    