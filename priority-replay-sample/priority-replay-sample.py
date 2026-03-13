import numpy as np 

def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """

    priorities = np.array(priorities)
    prior = priorities**alpha 
    prob = prior/np.sum(prior)
    N = len(prob)
    w = (N*prob)**(-beta)
    what = w/np.max(w)
    return [prob.tolist(), what.tolist()]