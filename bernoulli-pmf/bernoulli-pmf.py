import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """

    x = np.array(x)
    pmf = x*p + (1-x)*(1-p)
    mu = p
    var = p*(1-p)
    return pmf, mu, var