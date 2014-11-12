import numpy as np
import kmeans as KMeans


def initGaussianModel(data, K, maxIter=5, verbose=False):
    """
    Get initals parameters for the EM algorithm
    and return mean list, covarance list and weight list

    # PARAMS
    npArray data:   data to cluster
    int K           number of cluster to find
    int maxIter:    max number of iteration
    bool verbose:   make it talk

    # RETURN
    npArray means:  list of the intial means
    npArray cov:    list of covariance matrices
    npArray weight: list of intital weight
    """
    nbSamples, npFeatures = data.shape

    # Find the means values
    centerList = []
    bestIntraClusterVariance = 100000000
    for i in range(maxIter):
        members, centroids = KMeans.kmeans(data, K, verbose=verbose)
        icv = KMeans.getIntraClusterVariance(data, members, centroids)
        if icv < bestIntraClusterVariance:
            bestIntraClusterVariance = icv
            centerList = centroids

    # Initialize equivalent weighted mixture
    weights = np.ones(K) / K

    # Initialize diagonal covriant matrices
    cov = []
    for i in range(K):
        cov.append(np.diag(np.ones(npFeatures)))

    return centerList, cov, weights
