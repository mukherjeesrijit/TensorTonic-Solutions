import numpy as np

def percentiles(x, q):
    
    """
    Compute percentiles using linear interpolation.
    """

    x,q = map(np.array, [x,q])
    x = np.sort(x)
    n = len(x)
    virtual_id = q*(n-1)/100
    min_id = np.floor(q*(n-1)/100).astype(int) 
    min_id = np.clip(min_id, 0, n-2)
    alpha =  virtual_id - min_id
    result = (1-alpha)*x[min_id] + (alpha)*x[min_id+1]
    # result = np.percentile(x, q, method='linear')

    return result
    
    