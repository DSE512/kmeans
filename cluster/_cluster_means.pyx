cimport cython
import numpy as np
cimport numpy as np

ctypedef np.float64_t float64_t
ctypedef np.float32_t float32_t
ctypedef np.int32_t int32_t

ctypedef fused quantize_type:
    float32_t
    float64_t

# When the number of features is less than this number,
# switch back to the naive algorithm to avoid high overhead.
DEF NFEATURES_CUTOFF=5

# Initialize the NumPy C API
np.import_array()


@cython.cdivision(True)
cdef np.ndarray _update_cluster_means(quantize_type *obs, int32_t *labels,
                                      quantize_type *cb, int nobs, int nc, int nfeat):
    """Update k-means cluster means

    Args:
        obs : quantize_type*
            The pointer to the observation matrix.
        labels : int32_t*
            The pointer to the array of the labels (codes) of the observations.
        cb : quantize_type*
            The pointer to the new code book matrix.
        nobs : int
            The number of observations.
        nc : int
            The number of centroids (codes).
        nfeat : int
            The number of features of each observation.

    Returns: 
        has_members : ndarray
            A boolean array indicating which clusters have members.
    """
    cdef np.npy_intp i, j, cluster_size, label
    cdef quantize_type *obs_p
    cdef quantize_type *cb_p
    cdef np.ndarray[int, ndim=1] obs_count

    # Calculate the sums the numbers of obs in each cluster
    obs_count = np.zeros(nc, np.intc)
    obs_p = obs
    for i in range(nobs):
        label = labels[i]
        cb_p = cb + nfeat * label

        for j in range(nfeat):
            cb_p[j] += obs_p[j]

        # Count the obs in each cluster
        obs_count[label] += 1
        obs_p += nfeat

    cb_p = cb
    for i in range(nc):
        cluster_size = obs_count[i]

        if cluster_size > 0:
            # Calculate the centroid of each cluster
            for j in range(nfeat):
                cb_p[j] /= cluster_size

        cb_p += nfeat

    # Return a boolean array indicating which clusters have members
    return obs_count > 0


def update_cluster_means(np.ndarray obs, np.ndarray labels, int nc):
    """The update-step of K-means. 

    Calculate the mean of observations in each cluster.

    Args:
        obs : ndarray
            The observation matrix. Each row is an observation. Its dtype must be
            float32 or float64.
        labels : ndarray
            The label of each observation. Must be an 1d array.
        nc : int
            The number of centroids.
    Returns:
        cb : ndarray
            The new code book.
        has_members : ndarray
            A boolean array indicating which clusters have members.

    Notes:
        The empty clusters will be set to all zeros and the corresponding elements
        in `has_members` will be `False`. The upper level function should decide
        how to deal with them.
    """
    cdef np.ndarray has_members, cb
    cdef int nfeat

    # Ensure the arrays are contiguous
    obs = np.ascontiguousarray(obs)
    labels = np.ascontiguousarray(labels)

    if obs.ndim == 1:
        nfeat = 1
        cb = np.zeros(nc, dtype=obs.dtype)
    elif obs.ndim == 2:
        nfeat = obs.shape[1]
        cb = np.zeros((nc, nfeat), dtype=obs.dtype)
    else:
        raise ValueError('ndim different than 1 or 2 are not supported')

    if obs.dtype.type is np.float32:
        has_members = _update_cluster_means(<float32_t *>obs.data,
                                            <int32_t *>labels.data,
                                            <float32_t *>cb.data,
                                            obs.shape[0], nc, nfeat)
    elif obs.dtype.type is np.float64:
        has_members = _update_cluster_means(<float64_t *>obs.data,
                                            <int32_t *>labels.data,
                                            <float64_t *>cb.data,
                                            obs.shape[0], nc, nfeat)

    return cb, has_members
