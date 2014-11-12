import random
import numpy as np


def distanceWithPoint(data, point):
    """
    Return an array of distance between the point and each
    cell of the array
    """
    nbSample, nbFeatures = data.shape
    dist = np.zeros((nbSample, 1))

    for i in range(nbSample):
        tmp = 0.0
        for j in range(nbFeatures):
            tmp += (data[i][j] - point[j])**2
        dist[i] = tmp
    return np.sqrt(dist)


def getIntraClusterVariance(data, members, cent):
    """
    Return the intra cluster variance

    # PARAMS
    npArray data:   data
    int[] members:  membership number to each cluster
    npArray cent:   list of centroids

    # RETURN
    int var: variance
    """
    variance = 0.0
    for i in range(cent.shape[0]):
        consideredSample = data[members.view(np.ndarray).ravel() == i, :]
        variance += np.sum(distanceWithPoint(consideredSample, cent[i, :]))
    return variance


def findMembers(data, centroids):
    """
    Assign each data sample to the closest centroid

    # PARAMS
    narray data:        sample data
    narray centroids:   existing centroids

    # RETURN
    narray members:     index of the cluster member for each data sample
    """
    members = np.zeros((data.shape[0], 1))
    for (index, x) in enumerate(data):
        cent = min([(i[0], np.linalg.norm(x-centroids[i[0]]))
                   for i in enumerate(centroids)],
                   key=lambda t: t[1])[0]
        members[index] = cent
    return members


def updateCenters(data, members, K):
    """
    Find the new mean for each centroid and return the list of them

    # PARAMS
    npArray data:       sample data
    int K:              number of cluster
    npArray members:    index of the cluster member for each data sample

    # RETURN
    npArray centroids:  Extracted centroids (means) of each clusters
    """
    centroids = np.zeros([K, data.shape[1]])

    for i in range(K):
        matching = members.view(np.ndarray).ravel()
        centroids[i, :] = np.mean(data[matching == i, :], axis=0)
    return centroids


def kmeans(data, K, maxIter=10, verbose=False, interLoop=None):
    """
    Cluster all data samples into K cluster and return the centroids
    and the cluster assignment

    # PARAMS
    npArray data:   data to cluster
    int K           number of cluster to find
    int maxIter:    max number of iteration
    bool verbose:   make it talk
    func interLoop: function call after each loop
        interloop(data, membership, centroidList)

    # RETURN
    int[] mem:      membership number to each cluster
    npArray cent    list of centroids
    """

    def hasConverged(oldCentroids, centroids):
        return (set([tuple(a) for a in centroids])
                == set([tuple(a) for a in oldCentroids]))

    nbSamples, nbFeatures = data.shape
    centroids = random.sample(data, K)
    oldCentroids = random.sample(data, K)

    if verbose:
        print("Init KMean with :")
        print(centroids)

    for i in xrange(1, maxIter):
        members = findMembers(data, centroids)
        centroids = updateCenters(data, members, K)

        if interLoop is not None:
            interLoop(data, members, centroids)

        if hasConverged(oldCentroids, centroids):
            if verbose:
                print("Stop at iteration ", i)
            break

        oldCentroids = centroids

    if verbose:
        print("Final centroids :")
        print(centroids)
    return members, centroids
