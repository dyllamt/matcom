from numba import jit

import numpy as np

'''
this module implements functions for measuring feature vector similarity
between graph verticies. loop-based functions are just-in-time compiled with
numba, which results in fast code with limited memory overhead.

Note: all distances are stored at 64-bit floating point numbers
'''


@jit(nopython=True)
def pairwise_cos_distance(feature_vectors):
    '''
    pairwise cosine similarity function for verticies

    Args:
        feature_vectors (ndarray) M (num samples) x N (num features)

    Returns (ndarray) M x M array of bits
    '''
    M = feature_vectors.shape[0]
    N = feature_vectors.shape[1]
    D = np.empty((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            dot = 0.0
            idot = 0.0
            jdot = 0.0
            for k in range(N):
                tmp = feature_vectors[i, k] * feature_vectors[j, k]
                dot += tmp

                tmp = feature_vectors[i, k] * feature_vectors[i, k]
                idot += tmp

                tmp = feature_vectors[j, k] * feature_vectors[j, k]
                jdot += tmp
            D[i, j] = dot / idot / jdot
    return D


@jit(nopython=True)
def sub_pairwise_cos_distance(feature_vectors, sub_vectors):
    '''
    pairwise cosine similarity for a subset of verticies

    Args:
        feature_vectors (ndarray) M (samples) x N (features)
        sub_vectors (ndarray) m (samples) x N (features), where m <= M

    Returns (ndarray) M x m array of bits
    '''
    M = feature_vectors.shape[0]
    N = feature_vectors.shape[1]
    m = sub_vectors.shape[0]
    D = np.empty((M, m), dtype=np.float64)
    for i in range(M):
        for j in range(m):
            dot = 0.0
            idot = 0.0
            jdot = 0.0
            for k in range(N):
                tmp = feature_vectors[i, k] * sub_vectors[j, k]
                dot += tmp

                tmp = feature_vectors[i, k] * sub_vectors[i, k]
                idot += tmp

                tmp = feature_vectors[j, k] * sub_vectors[j, k]
                jdot += tmp
            D[i, j] = dot / idot / jdot
    return D


@jit(nopython=True)
def pairwise_squared_distance(feature_vectors):
    '''
    pairwise squared distance function for verticies

    Args:
        feature_vectors (ndarray) M (num samples) x N (num features)

    Returns (ndarray) M x M array of bits
    '''
    M = feature_vectors.shape[0]
    N = feature_vectors.shape[1]
    D = np.empty((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = feature_vectors[i, k] - feature_vectors[j, k]
                d += tmp
            D[i, j] = np.sqrt(d)
    return D


@jit(nopython=True)
def sub_pairwise_squared_distance(feature_vectors, sub_vectors):
    '''
    pairwise squared distance function for a subset of verticies

    Args:
        feature_vectors (ndarray) M (samples) x N (features)
        sub_vectors (ndarray) m (samples) x N (features), where m <= M

    Returns (ndarray) M x m array of bits
    '''
    M = feature_vectors.shape[0]
    N = feature_vectors.shape[1]
    m = sub_vectors.shape[0]
    D = np.empty((M, m), dtype=np.float64)
    for i in range(M):
        for j in range(m):
            d = 0.0
            for k in range(N):
                tmp = feature_vectors[i, k] - sub_vectors[j, k]
                d += tmp
            D[i, j] = np.sqrt(d)
    return D
