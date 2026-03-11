import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """

    memory = {}
    output = []
    for str in tokens:
        if str in memory:
            memory[str] += 1
        else: 
            memory[str] = 1
    print(memory)
    for w in vocab:
        output.append(int(memory.get(w, 0)))

    return np.array(output, dtype = int)
        