# GaussianMixtureModel

Simple binary classification using EM algorithm

## Usage

```
pip install requirements.txt
python main.py <training file> <unlabeled file> <output file>
```

## Optional parameters

* **-v --verbose**: Make it speak
* **-m --initMethod**: Select a specific method to initialize the EM algorithm. Available methods: *random*, *K-Means*, *Guess*. (default: *Guess*)
* **-K --nbCluster**: Required to use the *random* and *K-Means* init method. Execute the EM algorithm with a specific number of clusters. (default: 4)
* **-d --dimensions**: Number of dimensions to consider. (default: 2) 


## Data Format
Each line of the data file represent a sample in the below format:

```
Feature-Dim1 Feature-Dim2 [...] Feature-N  Class-Label
```
