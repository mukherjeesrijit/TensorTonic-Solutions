import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    X = np.array(X)
    X_c = X-np.mean(X,axis =0)
    cov = (X_c.T @ X_c)/len(X)
    std = np.std(X, axis = 0, keepdims = True)
    sigma = std .T @ std 
    R = cov/sigma
    return R