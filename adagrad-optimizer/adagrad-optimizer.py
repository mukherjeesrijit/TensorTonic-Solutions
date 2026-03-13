import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """

    w,g,G = map(lambda x: np.array(x, dtype = float), [w,g,G])
    G += g**2
    w -= lr*g*((np.sqrt(G+eps))**-1)
    return w, G