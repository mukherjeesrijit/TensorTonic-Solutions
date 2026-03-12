import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """

    Z1, Z2 = map(lambda x: np.array(x, dtype = float), [Z1,Z2])
    S = (Z1 @ Z2.T)/temperature
    max = np.max(S, axis = 1, keepdims = True)
    log = np.diag(S) - max - np.log(np.sum(np.exp(S-max),axis = 1))
    result = -np.mean(log)
    return result
    