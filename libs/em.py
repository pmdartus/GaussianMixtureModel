import numpy as np
import kmeans as KMeans


def pdf(data, mean, cov):
    """
    Return the probability density funct of sample data

    algo link: http://www.wikiwand.com/fr/Loi_normale_multidimensionnelle

    # PARAMS
    npArray data:   sample datas
    npArray mean:   list of means
    npArray cov:    list of covariance matrices

    # RETURN
    npArray proba:  probability for each data
    """
    dim = len(mean)

    leftPart = 1 / np.sqrt(np.power(2*np.pi, dim) * np.linalg.det(cov))
    mDiff = data - mean
    expPart = - (1/2) * np.transpose(mDiff) * np.linalg.inv(cov) * (mDiff)
    return leftPart * np.exp(expPart)


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
