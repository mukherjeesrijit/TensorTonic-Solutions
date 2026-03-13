import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """

    x = np.array(x)
    mu = np.mean(x)
    n = len(x)
    var = (1/(n-1)) * np.sum((x-mu)**2)
    return var, np.sqrt(var)