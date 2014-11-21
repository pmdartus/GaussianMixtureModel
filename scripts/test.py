import sys
import subprocess
import re
import time
import csv

outputFile = 'data/output.txt'


def executeCmd(trainingData, testData, verbose=False, method=None):
    cmd = ['python', 'main.py', trainingData, testData, outputFile]
    if verbose:
        cmd.append('-v')
    if method is not None:
        cmd.append('-m')
        cmd.append(method)
    begin = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, err = p.communicate()
    end = time.time()
    return end - begin, out, err


def diffFile(f1, f2):
    diffCmd = ['diff', '-y', '--suppress-common-lines', f1, f2]
    diff = subprocess.Popen(diffCmd, stdout=subprocess.PIPE)
    countLine = subprocess.Popen(['wc', '-lc'],
                                 stdin=diff.stdout, stdout=subprocess.PIPE)
    output = countLine.communicate()[0]
    count = re.search(r'[0-9]+', output)
    return int(count.group(0))


def getFileLength(f):
    cat = subprocess.Popen(['cat', f], stdout=subprocess.PIPE)
    countLine = subprocess.Popen(['wc', '-lc'],
                                 stdin=cat.stdout, stdout=subprocess.PIPE)
    output = countLine.communicate()[0]
    count = re.search(r'[0-9]+', output)
    return int(count.group(0))


def main(trainingData, testData, csv):
    nbSamples = getFileLength(testData)
    # methods = ["Guess", "Random", "K-Means"]
    methods = ["Guess"]
    for m in methods:
        print "\n#", m
        for i in range(4):
            print".",
            res = executeCmd(trainingData, testData, method=m)
            if res[2] is not None:
                print res[2]
            nbDiff = diffFile(testData, outputFile)
            accuracy = (float(nbDiff) / nbSamples)*100
            row = [m, accuracy, res[0]]
            csv.writerow(row)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception("Specify training and test file")

    trainingData = sys.argv[1]
    testData = sys.argv[2]
    with open('res_test.csv', 'wb') as csvFile:
        csvWriter = csv.writer(csvFile, delimiter=' ')
        main(trainingData, testData, csvWriter)
