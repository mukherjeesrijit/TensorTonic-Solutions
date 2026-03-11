import numpy as np 

def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """

    values, weights = map(np.array, [values, weights])
    
    delta = len(values)-len(weights)
    result = np.zeros(len(values)-len(weights)+1)
    for i in range(len(values)-len(weights)+1):
        print(values[i:i+len(weights)])
        print(weights)
        result[i] = np.dot(values[i:i+len(weights)],weights)/np.sum(weights)
    
    #result = np.convolve(values, weights[::-1], mode='valid')/np.sum(weights)
    return result.tolist()



    
        
        