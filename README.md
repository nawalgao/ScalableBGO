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


I am currently working on Neural Networks with last layer as a Bayesian layer. However, I am facing few difficulties related to it. These problems are as follows:

1. Uncertainity quantification in case of LLBNN seems to be dependent on the number of epochs I am running. Is it the problem with the model as a whole or is there a bug in my code?

Right now, here's the strategy I am using when concerned with LLBNN model :

1. Load the training data.
2. Fit a neural network (3 hidden layers, 50 hidden nodes for each layer) with backprop.
3. Instead of using the linear activation function over the outputs of the last layer of the neural network, save this output in the form of variable ll_out.
4. Use this last layer output as sort of an adaptive basis to train Bayesian Automatic Relevance Determination model. 
5. Update the prior, get the posterior distribution over the weights associated with adaptive basis.
6. Check the posterior predictive distribution.
7. Use PPD to calculate EI and drive the elicitation.




