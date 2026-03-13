import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """

    a,b,y = map(np.array, [a,b,y])
    a,b = map(np.atleast_2d,[a,b])
    d = a-b
    print(a)
    print(b)
    print(d)
    d = np.linalg.norm(a-b,axis = 1)
    print(d)
    print(y)
    result = y*(d**2) + (1-y)*(np.maximum(0,margin-d)**2)
    print(result.shape)
    if reduction == "mean":
        return np.mean(result)
    else:
        return np.sum(result)