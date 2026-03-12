import math 
import numpy as np 
from numpy.lib.stride_tricks import sliding_window_view as s_win

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """

    if len(candidate) == 0:
        return 0

    vocab = list(set(candidate+reference))
    vocab = {word:idx for idx, word in enumerate(vocab)} # vocab dictionary

    def encode(lst, vocab): # change lst to numerical
        for i in range(len(lst)):
            lst[i] = vocab[lst[i]]
        return lst 

    candidate = np.array(encode(candidate, vocab))
    reference = np.array(encode(reference, vocab))
    print(candidate)
    
    def bp(candidate, reference): # brevity penalty
        r = len(reference)
        c = len(candidate)
        return (1 if c >= r else math.exp(1-r/c)) if c > 0 else 0

    bp = bp(candidate,reference)
        
    def log_p_ngram(k, lst1, lst2): # k-gram (candidate, reference)
        min = np.min([len(lst1),len(lst2)])
        ng1 = s_win(lst1[:min], window_shape = k)
        ng2 = s_win(lst2[:min], window_shape = k)
        num = np.sum((ng1 == ng2).all(axis = 1))
        den = len(lst1)-k+1
        print(num)
        print(den)
        if den == 0 or num == 0:
            return 0.0
        return np.log(num/den)

    ngrams = np.arange(max_n)+1 # range of n_grams
    logp = [log_p_ngram(k, candidate, reference) for k in ngrams]
    bleu = bp * np.exp(np.mean(logp))

    return bleu