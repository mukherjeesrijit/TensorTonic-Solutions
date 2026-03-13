import numpy as np
from collections import Counter

def mean_median_mode(x):
    
    """
    Compute mean, median, and mode.
    """

    mean = np.mean(x)
    median = np.median(x)
    c = Counter(x)
    mode = max(c, key = c.get)
    
    return (mean, median, mode)