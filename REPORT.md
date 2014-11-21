# Gaussian mixture model estimation via Esitimation maximization

## Initialization

The result of the Expectation Maximization is tighlty bound to the initial state. The intial choise of the parameters for each components of the gaussian mixture, will have import consequence on the accuracy of the trainned model. Because the log-likelyhood is not most of the time a convex function, the algorithm can converge to some local-minimum. We discuss in this part what are the differents options offer to improve the accuracy of the gaussian mixuture model.

Because no informations are provided about the data set, the best way to initialize the model is to set the same weight and a identity matrix as variance-covariance matrix for each component of the mixture.
However the mean of each component can be estimated before executing the EM algorithm.

Two ways of choosing the means will be describe :
* **Random**: Choose randomly K samples as intial means
* **K-Means**: Use K-Means for create K clusters of samples and use the centroid of each as intial means

However we have assumed so far that the number of components (K) for the gaussian mixture is already known. A simple scatter plot on the training data allow to distinguish 4 components for each mixture. We will discuss in the next part how to estimate K.

## K Estimation

Getting the most optimal number of cluster out of a data set is a wellknow problem, an many algorithm already exist in order to solve it. In 2004, Pham et al. proposed a straight foreward algorithm wich has been inplemanted in this project. The algorithm simulates K-Means for several K and estimate the best K by comparing the improvement of the partitionning between two consecutive K.

This algorithm also return the centroids generated for the best K. Those centroids can be used to initialize the EM algorithm.

## Results

In order to test the script, we train the model with 4000 labelled samples (data/train.txt) generating 2 gaussian mixtures models. The trainned models are tested against 400 samples (data/dev.txt).
Each

## Further Work

## Conclusion
