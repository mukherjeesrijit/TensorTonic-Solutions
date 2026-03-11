import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    
    x1, x2 = map(np.array, [x1, x2])
    cosine = np.dot(x1,x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
    if label == 1:
        L = 1 - cosine
    else:
        L = max(0,cosine-margin)
    return L