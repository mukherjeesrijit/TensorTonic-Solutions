import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    
    X, y = map(np.array, [X, y]) 
    y = y[:,np.newaxis]
    
    w = np.zeros((X.shape[1],1))
    b = np.zeros(1)
    ones = np.ones((X.shape[0],1))

    def error(X,w,b,ones):
        return (_sigmoid(X@w + b*ones) - y)    
    
    def grad_w(X,w,b,ones):
        n = X.shape[0]
        err = error(X,w,b,ones)
        # print(X.T.shape)
        # print(err.shape)
        return 2/n * X.T @ err

    def grad_b(X,w,b,ones):
        n = X.shape[0]
        err = error(X,w,b,ones)
        return 2/n *np.sum(err)

    for _ in range(steps):
        # print(w.shape)
        # print(grad_w(X,w,b,ones).shape)
        # print(b.shape)
        # print(grad_b(X,w,b,ones).shape)
        w = w - lr * grad_w(X,w,b,ones)
        b = b - lr * grad_b(X,w,b,ones)

    return w.flatten(), b.item() 