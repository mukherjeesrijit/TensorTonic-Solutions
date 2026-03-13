import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    
    def pmf(k):
     return comb(n,k) * p**k * (1-p)**(n-k)

    pmf_k = pmf(k)
    cdf_k = np.sum([pmf(i) for i in range(k+1)])

    return (pmf_k, cdf_k)
    