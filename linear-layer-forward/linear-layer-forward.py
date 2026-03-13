import numpy as np 
def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    X,W,b = map(np.array, [X,W,b])
    return (X@W + b).tolist()