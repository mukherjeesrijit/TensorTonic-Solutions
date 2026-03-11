import numpy as np

def dropout(x, p=0.5, rng=None):
    
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    
    n = len(x)
    x = np.array(x)

    if rng: 
        r = rng.random(x.shape)
        print(r)
    else:    
        r = np.random.random(x.shape)
        print(r)

    pattern = np.where(r>=1-p,0,1)/(1-p)
    output = pattern * x
    
    return (output, pattern)