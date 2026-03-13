import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """

    p,y = map(np.array, [p,y])
    p,y = map(np.atleast_2d, [p,y])
    num = 2 * np.einsum('ij,ij->i',p,y) + eps
    den = np.sum(p + y, axis = 1) + eps
    loss = np.mean(num/den,axis = 0)
    return 1-loss