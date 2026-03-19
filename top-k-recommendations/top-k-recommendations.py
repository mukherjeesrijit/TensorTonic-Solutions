import numpy as np 

def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """

    scores = np.array(scores, dtype = float)
    rated_indices = list(rated_indices)
    scores[rated_indices] = -np.inf
    new_arr = np.argsort(-scores,kind='stable')
    return new_arr[:k].tolist() 