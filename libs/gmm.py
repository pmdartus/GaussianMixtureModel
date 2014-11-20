import numpy as np
import random
import kmeans as KMeans


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
    def __init__(self, trainingData, label):
        self.trainingData = trainingData
        self.label = label
        self.d = trainingData.shape[1]
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

    def estimateK(self, maxK=10):
        """ Estimate the best K and the associated centroids
        This algorithm is based on the on the detection of any significant
        increasement in the intra cluster density

        http://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf

        Update self.K and self.means
        """
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

    def calculateClusterCoherence(self, K, Skm1=0, maxIter=3):
        """
        Return the result of the f(K) function with associated means
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

        if K == 1:
            fs = 1
        elif Skm1 == 0:
            fs = 1
        else:
            fs = Sk / (alphaK(K, self.d) * Skm1)
        return fs, Sk, means

    def getRandomInitMeans(self):
        """ Draw K random sample as initial means for the EM algo

        Update self.means
        """
        self.means = random.sample(self.trainingData, self.K)

    def getInitKMeans(self, nbIter=10):
        """ Use KMeans to guess intial values of means

        Update self.means
        """
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
        if self.means is None or self.cov is None:
            err = "Gaussian Mixture Model should be init before trained"
            raise Exception(err)
