import numpy as np
import csv


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


def write_file(filePath, data, labels):
    with open(filePath, 'w') as f:
        for i, sample in enumerate(data):
            sample = ["{0:.6f}".format(j) for j in data[i]]
            sample = " ".join(sample)
            print sample
            f.write(sample + "  " + labels[i] + "\n")
