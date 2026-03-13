import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """

    x = np.array(x)
    if (x.ndim == 1) and (len(x) == 1):
        return np.array([1])
    x = np.atleast_2d(x)
    max = np.max(x, axis = 1, keepdims = True)
    num = np.exp(x-max)
    # print(num)
    den = np.sum(num, axis = 1, keepdims = True)
    # print(den)
    return (num/den).squeeze()