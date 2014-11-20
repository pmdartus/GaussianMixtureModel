import csv
import argparse
from os import path

import numpy as np
import matplotlib.pyplot as plt

from libs.gmm import GaussianMixtureModel


def main():
    parser = argparse.ArgumentParser(
        description='Implement EM algorithm for GMM.'
    )
    parser.add_argument(
        'training',
        help='File containing the training data'
    )
    parser.add_argument(
        'file',
        help='File containing the data to be classified'
    )
    parser.add_argument(
        'output',
        help='File containing the labeled data'
    )
    parser.add_argument(
        '-d',
        '--dimension',
        help='Number of dimentional feature',
        required=False,
        default=2
    )
    parser.add_argument(
        '--initMethod',
        help='Election method for initials means `Random` or `K-Means`',
        required=False,
        default="K-Means"
    )
    parser.add_argument(
        '-v',
        '--verbose',
        help='Make it speak',
        action='store_true'
    )

    args = parser.parse_args()

    if not path.isfile(args.training) or not path.isfile(args.file):
        parser.error('Invalid file path')

    # Extract data from training file
    data, labels = read_file(args.training, args.dimension, True)
    dataStore = createDataStore(data, labels)

    # Execute Expectation maximization algorithm to each label data
    models = []
    for label in dataStore.keys():
        model = GaussianMixtureModel(dataStore[label], label,
                                     verbose=args.verbose)
        model.initModel(args.initMethod, K=4)
        model.train()
        models.append(model)
        if args.verbose:
            print "======================== \n"

    # Load unlabeled data
    data, expectedLabels = read_file(args.file, args.dimension, True)
    labels = affectLabelsToData(data, models)
    print labels


def createDataStore(data, labels):
    """Bin same labeled data in a dict and return it

    # Params
    data(Array)
    labels(Array)
    """
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
    """ Read file and return extracted data and labels

    # Params
    filepath (String):
        Relative path to the data file
    dimension (Integer):
        Number of dimension to extract from the file
    withlabel (Boolean) optional default=False:
        Extract the labels for each sample

    # Return
    data (Array nb*dimension):
        List of data sample
    labels (Array nb):
        list of labels for each sample
    """
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


def affectLabelsToData(data, models):
    """
    Return the list labels for each samples based on the provided model

    # Params
    data (Array)
    models (Array GaussianMixtureModel)
    """
    p_Xn = np.zeros((len(data), len(models)))
    for i, trainnedModel in enumerate(models):
        p_Xn[:, i] = trainnedModel.pdf(data)
    bestModel = np.argmax(p_Xn, axis=1)
    return [models[i].label for i in bestModel]


if __name__ == '__main__':
    main()
