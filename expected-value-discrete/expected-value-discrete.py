import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x,p = map(np.array, [x,p])
    if (p < 0).any() or np.sum(p) != 1:
        raise ValueError
    return np.dot(x,p)