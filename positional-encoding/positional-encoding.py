import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    num = np.arange(seq_len)
    f = lambda i: (base ** ((2*np.floor(i/2))/d_model))**-1
    den = f(np.arange(d_model))
    theta = np.outer(num,den)
    PE = np.ones((theta.shape))
    PE[:,0::2] = np.sin(theta[:,::2])
    PE[:,1::2] = np.cos(theta[:,1::2])
    return PE