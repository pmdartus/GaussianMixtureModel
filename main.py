import csv
import argparse
from os import path

import numpy as np
import matplotlib.pyplot as plt

import libs.em as EM


def main():
    parser = argparse.ArgumentParser(
        description='Implement EM algorithm for GMM.'
    )
    parser.add_argument(
        'training',
        help='file containing the training data'
    )
    parser.add_argument(
        'file',
        help='file containing the data to be classified'
    )
    parser.add_argument(
        '--dimension',
        help='number of dimentional feature',
        required=False,
        default=2
    )

    args = parser.parse_args()

    if not path.isfile(args.training) or not path.isfile(args.file):
        parser.error('Invalid file path')

    # Extract data from training file
    data, labels = read_file(args.training, args.dimension, True)
    dataStore = createDataStore(data, labels)

    # Execute Expectation maximization algorithm to each label data
    mixtureParams = []
    for k in dataStore:
        means, cov, weights = EM.initGaussianModel(dataStore[k], 4)
        params = EM.run(dataStore[k], means, cov, weights, 4)
        mixtureParams.append(params)
    printDataStore(dataStore)
    simulateData(mixtureParams)


def createDataStore(data, labels):
    """Bin same labeled data in a dict"""
    dataStore = dict()
    for pos, val in enumerate(data):
        dataLabel = labels[pos]
        if(dataLabel in dataStore.keys()):
            dataStore[dataLabel].append(val)
        else:
            dataStore[dataLabel] = [val]
    for label in dataStore:
        dataStore[label] = np.array(dataStore[label])
    return dataStore


def read_file(filepath, dimension, withlabel=False):
    """Read file and return extracted data and labels"""
    with open(filepath) as f:
        csv_reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        labels = []
        data = []
        for sample in csv_reader:
            features = [float(i) for i in sample[:dimension]]
            data.append(features)
            if withlabel:
                labels.append(sample[-1])
    return data, labels


def printDataStore(dataStore, centroids=None):
    """Print the dataStore on a console"""
    colors = ['blue', 'green', 'magenta', 'cyan']
    for k in dataStore:
        values = dataStore[k]
        x = values[:, 0]
        y = values[:, 1]
        c = colors.pop()
        plt.scatter(x, y, color=c, alpha=0.3)

    if centroids is not None:
        for c in centroids:
            plt.scatter(c[0], c[1], color="red")
    plt.show()


def simulateData(mixtureParams):
    numPoints = 200
    for params in mixtureParams:
        nbClusters = len(params[2])
        for i in range(nbClusters):
            MU = params[0][i]
            SIGMA = params[1][i]
            pointSet = np.random.multivariate_normal(MU, SIGMA, numPoints)
            plt.scatter(pointSet[:, 0], pointSet[:, 1])
    plt.show()

if __name__ == '__main__':
    main()
