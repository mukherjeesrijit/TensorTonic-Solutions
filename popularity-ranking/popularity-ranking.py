import numpy as np 

def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    
    items = np.array(items, dtype = float)
    num = np.prod(items, axis = 1) + min_votes*global_mean
    den = items[:,1]+min_votes
    result = num/den
    return result.tolist()