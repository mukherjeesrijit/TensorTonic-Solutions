import numpy as np 

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """

    # prove that micro f1 = #correct / N

    all = y_true+y_pred
    y_true, y_pred = map(np.array, [y_true, y_pred])
    C = np.sum(y_true==y_pred) #correct
    N = len(y_true)
    return C/N


    