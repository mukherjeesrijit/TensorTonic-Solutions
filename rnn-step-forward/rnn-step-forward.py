import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """

    x_t, h_prev, Wx, Wh, b = map(np.array, [x_t, h_prev, Wx, Wh, b])
    val = x_t @ Wx + h_prev @ Wh + b 
    return np.tanh(val)
    