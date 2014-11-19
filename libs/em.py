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
    det = np.linalg.det(cov)

    norm_const = 1 / np.sqrt(np.power(2 * np.pi, dim) * det)
    x_mu = data - mean
    inv = np.linalg.inv(cov)
    exp = -0.5 * x_mu.dot(inv.dot(np.transpose(x_mu)))
    return norm_const * np.exp(exp)


def logmulnormpdf(X, MU, SIGMA):
    """
    Evaluates natural log of the PDF for the multivariate Guassian distribution.

    Parameters
    ----------
    X : np array
        Inputs/entries row-wise. Can also be a 1-d array if only a
        single point is evaluated.
    MU : nparray
        Center/mean, 1d array.
    SIGMA : 2d np array
        Covariance matrix.

    Returns
    -------
    prob : 1d np array
        Log (natural) probabilities for entries in `X`.

    """

    mu = MU
    x = X.T
    sigma = np.atleast_2d(SIGMA)  # So we also can use it for 1-d distributions

    N = len(MU)
    ex1 = np.dot(np.linalg.inv(sigma), (x.T-mu).T)
    ex = -0.5 * (x.T-mu).T * ex1
    if ex.ndim == 2:
        ex = np.sum(ex, axis=0)
    K = -(N/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(SIGMA))
    return ex + K


def run(data, means, cov, weights, K, epsilon=1e-5, maxIter=10):
    """
    Run the EM algorithm and return the parameters when there is convergence
    or the max number of iteration hase been reached
    """
    old_LogL = 0
    LogL = 0
    for i in range(maxIter):
        means, cov, weights, LogL = EMSteps(data, means, cov, weights, K)
        if np.abs(old_LogL - LogL) < epsilon:
            break
        else:
            old_LogL = LogL
    return (means, cov, weights)


def EMSteps(data, means, cov, weights, K):
    """
    Run a single step of the EM algorithm return the updated parameters
    with the max likelihood
    """
    nbSamples, nbFeatures = data.shape
    log_pXn_mat = np.zeros((nbSamples, K))
    for i in range(K):
        tmp = logmulnormpdf(data, means[i], cov[i])
        log_pXn_mat[:, i] = tmp + np.log(weights[i])
    pMax = np.max(log_pXn_mat, axis=1)
    log_pXn = pMax + np.log(np.sum(np.exp(log_pXn_mat.T - pMax), axis=0).T)
    logL = np.sum(log_pXn)

    log_pNk = np.zeros((nbSamples, K))
    for i in range(K):
        log_pNk[:, i] = log_pXn_mat[:, i] - log_pXn

    pNk = np.e**log_pNk

    for i in range(K):
        means[i] = np.sum(pNk[:, i] * data.T, axis=1) / np.sum(pNk[:, i])
        cov[i] = np.dot(pNk[:, i] * (data - means[i]).T, data - means[i]) / np.sum(pNk[:, i])
        weights[i] = np.sum(pNk[:, i]) / nbSamples

    return (means, cov, weights, logL)


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

    return centerList, np.array(cov), weights
