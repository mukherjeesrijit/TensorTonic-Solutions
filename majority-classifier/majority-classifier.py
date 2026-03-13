import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train, X_test = map(np.array, [y_train, X_test])
    values, counts = np.unique(y_train, return_counts = True)
    #print(values[np.argmax(counts)])
    arr = np.empty(len(X_test))
    arr.fill(values[np.argmax(counts)])
    return arr