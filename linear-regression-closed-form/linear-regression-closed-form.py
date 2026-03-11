import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    X,y = map(np.array,[X,y])
    return np.linalg.inv((X.T @ X))@X.T@y