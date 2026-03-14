import numpy as np 

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """

    values = np.array(values)
    min = np.min(values)
    max = np.max(values)
    w  = (max-min)/num_bins
    if w==0:
        return np.zeros(len(values)).tolist()
    binf = lambda x: np.minimum(np.floor((x-min)/w),num_bins-1)
    binf = np.vectorize(binf)
    return binf(values).tolist()
