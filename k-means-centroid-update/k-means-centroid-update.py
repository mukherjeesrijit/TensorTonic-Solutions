import numpy as np 

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here

    points, assignments = map(np.array, [points, assignments])

    centers = np.zeros((k,points.shape[1]))
    counts = np.zeros(k)

    for i in range(points.shape[0]):
        centers[assignments[i]] += points[i]
        counts[assignments[i]] += 1

    print(centers/counts[:,np.newaxis])
    
    return (centers/counts[:,np.newaxis]).tolist()
    