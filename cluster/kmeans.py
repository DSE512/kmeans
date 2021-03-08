import numpy as np

from collections import deque
from scipy.spatial.distance import cdist
from cluster._cluster_means import update_cluster_means


def euclidean_distance(x1, x2):
    """Squared distance

    We are ignoring the sqrt here to since it
    does not affect k-means, and we gain a bit
    of speed. 

    Note:
        While k-means is unaffected by using the 
        squared distance, other clustering 
        algorithms, like hierarchical clustering,
        are affected. Instead, you should use the 
        regular Euclidean distance.
    """
    return np.sum(np.square(x1 - x2))


def initialize_centroids(data, k):
    """Pick random k points as initial centroids"""
    num_samples, _ = data.shape
    idx = np.random.choice(num_samples, size=k, replace=False)
    return data[idx]


def vector_quantize(data, centroids):
    dist = cdist(data, centroids)
    code = dist.argmin(axis=1)
    min_dist = dist[np.arange(len(code)), code]
    return code, min_dist


def kmeans(data, k, num_iter=10, threshold=1e-5):
    """"""
    minimal_distance = np.inf

    for _ in range(num_iter):
        centroids = initialize_centroids(data, k)
        centroids, distance = _kmeans(data, centroids, threshold=threshold)
        if distance < minimal_distance:
            best_centroids = centroids
            best_distance = distance

    return best_centroids, best_distance


def _kmeans(data, centroids, threshold=1e-5):
    diff = np.inf
    prev_dists = deque([diff], maxlen=2)
    while diff > threshold:
        code, min_dist = vector_quantize(data, centroids)
        prev_dists.append(min_dist.mean(axis=-1))

        codebook, has_members = update_cluster_means(
            data, code, centroids.shape[0]
        )

        codebook = codebook[has_members]
        diff = prev_dists[0] - prev_dists[1]

        return codebook, prev_dists[1]

