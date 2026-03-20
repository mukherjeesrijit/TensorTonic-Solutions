import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """

    # note that we are given a batch of vectors
    x = np.array(x) # transforming list into numpy array
    x = np.atleast_2d(x) # to make sure for batch 1 has rank 2
    # print(x.shape)
    x = x - np.max(x, axis = 1, keepdims = True) # max to reduce axis 1
    # print(x.shape)
    num = np.exp(x) # numerator = pointwise exponential for all values
    # print(num.shape)
    den = np.sum(num, axis = 1, keepdims = True) # denominator = summation to reduce axis 1
    # print(den.shape)
    if x.shape[0] == 1:
        return (num/den)[0]
    return (num/den)