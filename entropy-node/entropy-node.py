import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """

    # relative frequency -> probability vector p
    values, counts = np.unique(y, return_counts = True)
    p = counts/len(y)
    
    # - dot product of log (p) & p
    entropy = - np.dot(p,np.log2(p))

    return entropy