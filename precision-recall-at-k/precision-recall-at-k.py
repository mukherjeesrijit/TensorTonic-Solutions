import numpy as np

def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """

    precisionk = len(np.intersect1d(recommended[:k],relevant))/k
    recallk = len(np.intersect1d(recommended[:k],relevant))/len(relevant)
    return [precisionk,recallk]

    