import numpy as np 

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """

    points, centroids = map(np.array, [points, centroids])
    points = points[:,:,np.newaxis]
    centroids = centroids.T[np.newaxis,:,:]
    norms = np.linalg.norm((points - centroids),axis = 1)
    assignment = np.argmin(norms, axis = 1)
    return assignment.tolist()