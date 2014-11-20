import numpy as np
import random

import kmeans as KMeans
import em as EM


class GaussianMixtureModel(object):
    """ Used to train itself based on training data.
    Return the probability for unlabeled data to belong to the model

    # Attributes
    trainingData (npArray):
        Data to train the model
    label (String):
        Label of the model, used to partion the data
    d (Integer):
        Dimenstion of each samples of the dataSet
    means (npArray K*d):
        List of means for each gaussian
    cov (npArray K*d*d):
        List of variance-covariance matrices for each gaussian
    weights (npArray K):
        List of weights for each component of the mixture
    """
    def __init__(self, trainingData, label, verbose=False):
        self.trainingData = trainingData
        self.label = label
        self.d = trainingData.shape[1]
        self.verbose = verbose
        self.means = None
        self.cov = None
        self.weights = None

    def initModel(self, method, K=0, guessK=False):
        """ Init to model based on provided training data.
        It should be executed before before running the EM algorithm

        # Params
        method (String):
            Initial means election: `Random`, `K-Means`
        K (Integer):
            Number of features constituing the model
        guessK (boolean):
            Used Pham algorithm to estimate K

        Update self.K, self.means, self.cov and self.weights
        """
        if self.verbose:
            print "# INIT Gaussian Mixture Model", self.label

        # Check validity of parameters
        if K == 0 and guessK is False:
            raise Exception("Init with eather K set guessK to True")
        if K != 0:
            self.K = K
        else:
            self.estimateK()

        # Init means
        if self.means is None:
            if method == "Random":
                self.getRandomInitMeans()
            elif method == "K-Means":
                self.getInitKMeans()
            else:
                raise Exception("Not inplemented methods")

        # Init weights
        self.weights = np.ones(self.K) / self.K

        # Init covariance matrices
        cov = []
        for i in range(self.K):
            cov.append(np.diag(np.ones(self.d)))
        self.cov = np.array(cov)

    def estimateK(self, maxK=5):
        """ Estimate the best K and the associated centroids
        This algorithm is based on the on the detection of any significant
        increasement in the intra cluster density

        http://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf

        # Params:
        maxK (Integer) optional:
            Max number of possible clusters

        Update self.K and self.means
        """
        if self.verbose:
            print "    Guessing number of components of the mixture"

        ks = range(1, maxK)
        fs = np.zeros(len(ks))
        centroids = []

        fs[0], Sk, retCentroids = self.calculateClusterCoherence(1)
        centroids.append(retCentroids)

        for k in ks[1:]:
            fs[k-1], Sk, retCentroids = self.calculateClusterCoherence(
                k, Skm1=Sk)
            centroids.append(retCentroids)

        bestCentroids = centroids[fs.argmin()]
        self.K = len(bestCentroids)
        self.means = np.array(bestCentroids)

        if self.verbose:
            print "    Number of components for", self.label, ":", self.K

    def calculateClusterCoherence(self, K, Skm1=0, maxIter=3):
        """
        Return the result of the f(K) function with associated means

        # Params
        K (Integer):
            Considered number of different clusters
        SKm1 (Integer) optional:
            Previous intra cluster variance
        maxIter (Integer) optional default=3:
            Number of iteration to select best K-Means partition

        # Return
        fs (Integer):
            Evluation of the function f(K)
        Sk (Integer):
            Intra cluster variance for K clusters
        means (npArray K*d):
            Generated means via K-Means
        """

        def alphaK(k, Nd):
            if k == 2:
                return 1 - 3/(4*Nd)
            else:
                return alphaK(k-1, Nd) + (1 - alphaK(k - 1, Nd))/6

        data = self.trainingData

        means = None
        Sk = None
        for i in range(maxIter):
            members, centroids = KMeans.kmeans(data, K)
            trySk = KMeans.getIntraClusterVariance(data, members, centroids)
            if (Sk is None or trySk < Sk):
                Sk = trySk
                means = centroids

        if K == 1 or Skm1 == 0:
            fs = 1
        else:
            fs = Sk / (alphaK(K, self.d) * Skm1)
        return fs, Sk, np.array(means)

    def getRandomInitMeans(self):
        """ Draw K random sample as initial means for the EM algo

        Update self.means
        """
        if self.verbose:
            print "    Use `Random` method to init model"
        self.means = random.sample(self.trainingData, self.K)

    def getInitKMeans(self, nbIter=10):
        """ Use KMeans to guess intial values of means

        Update self.means
        """
        if self.verbose:
            print "    Use `K-Means` method to init model"
        centerList = []
        bestIntraClusterVariance = None
        data = self.trainingData
        K = self.K

        for i in range(nbIter):
            members, centroids = KMeans.kmeans(data, K)
            icv = KMeans.getIntraClusterVariance(data, members, centroids)
            # If intracluster variance is lower update centroids
            if (bestIntraClusterVariance is None
                    or icv < bestIntraClusterVariance):
                bestIntraClusterVariance = icv
                centerList = centroids
        self.means = np.array(centerList)

    def train(self):
        """ Train the model based on the provided data """
        if self.verbose:
            print "# TRAINING model", self.label

        if self.means is None or self.cov is None:
            err = "Gaussian Mixture Model should be init before trained"
            raise Exception(err)

        params = EM.run(self.trainingData, self.means, self.cov,
                        self.weights, self.K)
        self.means = params[0]
        self.cov = params[1]
        self.weights = params[2]

    def gaussianPdf(self, data, i):
        """Evaluate the pdf of data to belong for the ith component
        of the mixture

        # Params:
        data (npArray)
        i (Integer):
            Represent the ith component of the mixture

        # Return:
        proba (npArray):
            Proba for each data
        """
        mu = self.means[i]
        x = np.array(data).T
        sigma = np.atleast_2d(self.cov[i])

        ex1 = np.dot(np.linalg.inv(sigma), (x.T-mu).T)
        ex = -0.5 * (x.T-mu).T * ex1
        ex = np.sum(ex, axis=0)
        K = 1 / np.sqrt(np.power(2*np.pi, self.d) * np.linalg.det(sigma))
        return K*np.exp(ex)

    def pdf(self, data):
        """Evaluate the pdf of data to belong to the mixture

        # Params:
        data (npArray)

        # Retrun:
        proba (npArray)
        """
        proda = None
        for i in range(self.K):
            weight = self.weights[i]
            probaToAdd = self.gaussianPdf(data, i) * weight
            if proda is None:
                proda = probaToAdd
            else:
                proda += probaToAdd
        return proda
