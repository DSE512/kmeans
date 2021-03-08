import numpy as np

from . import _cluster_meanj


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
    dist = euclidean_distance(data, centroids)
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
    while diff > threshold:
        #TODO(Todd): Figure out how to handle diff here.
        prev_diff = diff
        code, min_dist = vector_quantize(data, centroids)

        codebook, has_members = _cluster_means.update_cluster_means(
            data, obs_code, centroids.shape[0]
        )

        codebook = code_book[has_members]
        diff = diff - prev_diff

        return codebook, prev_avg_dists[1]

