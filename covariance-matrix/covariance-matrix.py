import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.atleast_2d(X)
    n = len(X)
    if n == 1:
        return None
    Xcen = X - np.mean(X, axis = 0)
    cov = (Xcen.T @ Xcen)/(n-1)

    return cov