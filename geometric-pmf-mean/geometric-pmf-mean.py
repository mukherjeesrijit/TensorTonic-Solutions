import numpy as np

def geometric_pmf_mean(k, p):
    
    """
    Compute Geometric PMF and Mean.
    """

    k = np.array(k)
    pmf_k = (1-p)**(k-1) * p
    mean = 1/p
    
    return (pmf_k, mean)