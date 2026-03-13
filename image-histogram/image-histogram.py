import numpy as np 

def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """

    image = np.array(image)
    bins = np.zeros(256)
    vals, cnts = np.unique(image.reshape(-1),return_counts = True)
    bins[vals] = cnts
    return bins.tolist()