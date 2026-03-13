import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """

    x = np.array(x)
    f = lambda x: 0.5*x*(1+math.erf(x/math.sqrt(2)))
    return np.vectorize(f)(x)
