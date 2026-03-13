import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """

    p,q = map(np.array, [p,q])
    result = p/q
    result = np.log(result)
    result = np.dot(result,p)
    return result