import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    
    rater1,rater2 = map(np.array, [rater1,rater2])
    n = len(rater1)
    p0 = np.sum(rater1==rater2)/n
    values1, counts1 = np.unique(rater1, return_counts = True)
    values2, counts2 = np.unique(rater2, return_counts = True)
    pe = np.sum((counts1*counts2)/(n**2))  
    if pe == 1:
        return 1
    k = (p0-pe)/(1-pe)
    return k