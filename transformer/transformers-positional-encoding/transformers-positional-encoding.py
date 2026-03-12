import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    base=10000.0
    num = np.arange(seq_length)
    f = lambda i: base ** ((2*np.floor(i/2))/d_model)
    den = f(np.arange(d_model))
    PE = np.outer(num,den**-1)
    PE[:,0::2] = np.sin(PE[:,::2])
    PE[:,1::2] = np.cos(PE[:,1::2])
    return PE

    
    
    