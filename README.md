# ScalableBGO

As of now, ScalableBGO needs the following dependencies
1. gpflow (https://github.com/GPflow/GPflow)

GPflow implements modern Gaussian process inference for composable kernels and likelihoods. The online user manual contains more details. The interface follows on from GPy, for more discussion of the comparison see this page.

GPflow uses TensorFlow for running computations, which allows fast execution on GPUs, and uses Python 3.5 or above.

Manually install it as follows:
git clone https://github.com/GPflow/GPflow.git
cd GPflow
pip install .



Right now, we have the following aquisition functions (which selects multiple points - parallelizable) implemented, example of which can be checked out in scripts > model_comparisons.py

1. GP-MLE + Snoek Fantasy EI
2. GP-MLE + Constant Liar Strategy 
3. GP-MLE with crude user defined EI
4. Neural Network with last layer Bayesian (still under progress)



