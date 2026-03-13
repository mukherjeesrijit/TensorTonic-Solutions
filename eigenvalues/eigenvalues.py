import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """

    if (matrix == []) or not isinstance(matrix[0],list):
        return None
    
    elif len(matrix) != len(matrix[0]):
        return None

    else:
        matrix = np.array(matrix)        
        vals, vecs = np.linalg.eig(matrix)
        # vecs[:i] ~ vals[i]
        val_idx = np.lexsort((vals.real, vals.imag))
        # print(val_idx)
        return vals[val_idx]    
    