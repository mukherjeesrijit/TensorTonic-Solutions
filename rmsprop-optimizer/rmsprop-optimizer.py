import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    
    w,g,s = map(lambda x: np.array(x,dtype = float),[w,g,s])
    s = beta * s + (1-beta)*(g**2)
    w = w - lr * g / (np.sqrt(s+eps))
    return (w,s)