import numpy as np

def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    
    # if len(sentences) >= 1 and not len(sentences):
    #     all = sentences
    all = []
    for x in sentences:
        all += x 

    vocab = {}
    for word in all:
        if word in vocab:
            vocab[word] += 1
        else: 
            vocab[word] = 1

    return vocab 