import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """

    n = len(X)
    X = np.array(X)
    
    # centerizing the data for the row 
    X = X - np.mean(X, axis = 0)

    # Covariance Matrix
    C = (n-1)**(-1) * X.T @ X

    # Eigenvectors & top k components
    eigval, eigvec = np.linalg.eig(C)
    top_k = np.argsort(eigval)[::-1][:k] # top k eigval indices 
    pc_k = eigvec.T[top_k]

    # Projection of Data to the Eigenvectors
    X_proj = X @ pc_k.T

    return X_proj