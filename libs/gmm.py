import numpy as np
import kmeans as KMeans
import em as EM


class GaussianMixtureModel(object):
    """
    Used to train itself based on training data.
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
    """
    def __init__(self, trainingData, label):
        self.trainingData = trainingData
        self.label = label
        self.d = trainingData.shape[1]
        self.means = None
        self.cov = None
        self.weights = None

    def initModel(self, method, K=0, guessK=False):
        """
        Init to model based on provided training data.
        It should be executed before before running the EM algorithm

        # Params
        method (String):
            Initial means election: `Random`, `K-Means`
        K (Integer):
            Number of features constituing the model
        guessK (boolean):
            Used Pham algorithm to estimate K
        """
        # Check validity of parameters
        if K == 0 and guessK is False:
            raise Exception("Init with eather K set guessK to True")
        if K != 0:
            self.K = K
        else:
            self.K, self.means = KMeans.estimateK(self.trainingData)

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

    def getRandomInitMeans(slef):
        """
        Draw K random sample as initial means for the EM algo
        """
        pass

    def getInitKMeans(self):
        """
        Use KMeans to guess intial values of Means
        """
        pass

    def train(self):
        """
        Train the model based on the prvided data
        """
        if self.means is None or self.cov is None:
            err = "Gaussian Mixture Model should be init before trained"
            raise Exception(err)
