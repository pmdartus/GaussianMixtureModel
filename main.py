import argparse
from os import path
import numpy as np

from libs.gmm import GaussianMixtureModel
from libs.helpers import read_file, write_file, createDataStore


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
        type=int,
        default=2
    )
    parser.add_argument(
        '-m',
        '--initMethod',
        help='Election method for initials means `Random` `K-Means` or `Guess`',
        required=False,
        default="Guess"
    )
    parser.add_argument(
        '-K',
        '--nbCluster',
        help='Nb components for each mixtures',
        required=False,
        type=int,
        default=4
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
        if args.initMethod == "Guess":
            model.initModel(args.initMethod, guessK=True)
        else:
            model.initModel(args.initMethod, K=args.nbCluster)
        model.train()
        models.append(model)
        if args.verbose:
            print "======================== \n"

    # Load unlabeled data
    data, expectedLabels = read_file(args.file, args.dimension, True)
    labels = affectLabelsToData(data, models)

    # Write it back!
    write_file(args.output, data, labels)


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
