import numpy as np
import neural_net
import bayesian_linear_regressor
import sys
sys.path.append('../')
from misc import normalization
from learning_objective import gaussian_process


class LLBayesianNN(object):
    def __init__(self, X, y, rng = 1,
                 normalize_input = True, normalize_output = True,
                 normalize_output_lr = False,
                 num_epochs = 50000, batch_size = 50,
                 n_units_1 = 50, n_units_2 = 50, n_units_3 = 50,
                 lr_intercept = False):
        """
        Deep Neural Networks with last layer as a bayesian regressor layer
        This module performs Bayesian Linear Regression with basis function extracted
        from a feed forward neural network.
        J. Snoek paper on Scalable Bayesian Optimization

        Parameters:
        ------------
        X : np.ndarray(N, D)
        Input data points. Dimensionality of X is (N , D),
        with N as the number of datapoints and D is the number of features
        y : np.ndarray(N,)
        Corresponding target values
        """
        self.X = X
        self.y = y
        self.rng = rng
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.normalize_output_lr = normalize_output_lr
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Hidden layers configuration
        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3

        # Bayesian Linear Regression Configuration
        self.lr_intercept = lr_intercept

    def train(self):
        """
        Train the LL_Bayesian_NN model using the available training dataset
        This will consist of roughly two important tasks
        First, NN will perform the task of feature extraction 
        Second, Bayesian Linear Regressor will condition the above obtained features.
        This BLR layer is going to help us in making predictions and quantify the epistemic uncertainity
        associated with our predictions.
        """
        # Train the neural network for feature extraction
        NN = neural_net.NeuralNet(self.X, self.y, self.rng,
                                  self.normalize_input, self.normalize_output)
        NN.train_nn(self.num_epochs, self.batch_size,
                   self.n_units_1, self.n_units_2, self.n_units_3)
        ll_out = NN.extract_last_layer_nn_out(self.X)

        self.NN = NN

        # Bayesian Linear Regression (with Automatic Relevance determination to infer hyperparameters)
        LinearRegressor = bayesian_linear_regressor.BayesianARD(ll_out, self.y,
                                                                self.normalize_output_lr,
                                                                self.lr_intercept)
        m, S, sigma, alpha = LinearRegressor.train()

        #self.m = m
        #self.S = S
        #self.sigma = sigma
        #self.alpha = alpha

        self.LinearRegressor = LinearRegressor
        
        np.savetxt('../nn_results/ll_out_E' + str(self.num_epochs) + 'N' + str(self.X.shape[0]) + '.txt', ll_out)

        #return m, S, sigma, alpha

    def test(self, X_test):
        """
        Predict the objective function value for new testing set X_test

        Parameters
        ---------------
        X_test : testing set (N x D)
        """

        if self.normalize_input:
            X_test = normalization.zero_mean_unit_var_normalization(X_test)[0]

        ll_out_test = self.NN.extract_last_layer_nn_out(X_test)
        #print ('LL out shape')
        #print (ll_out_test.shape)
        mean, sd = self.LinearRegressor.test(ll_out_test)
        
        return mean, sd


if __name__ == '__main__':
#    X_train = np.linspace(0, 80, 100).reshape(-1, 1)
#    Y_train = 5 * X_train
    
    X_train = np.linspace(-1, 1, 20)[:,None]
    Y_train = gaussian_process.gaussian_process(X_train)[:,0]
    X_test = np.linspace(np.min(X_train), np.max(X_train), 200)[:,None]

    LLBNN = LLBayesianNN(X_train, Y_train, normalize_output_lr = True,
                         num_epochs = 4000)
    LLBNN.train()
    print('Training is done')
    
    # Testing on training data itself
    r = LLBNN.test(X_test)
    #print ('Mean')
    #print (r[0])
    #print ('SD')
    #print (r[1].shape)
    #print ('Lower')
    #print (r[3].shape)
    #print ('Upper')
    #print (r[2].shape)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('talk')
    sns.set_style('white')
    plt.figure()
    plt.plot(X_test, r[0])
    r1 = r[0] + 2*r[1]
    r2 = r[0] - 2*r[1]
    plt.fill_between(X_test.flatten(), r1, r2, color=sns.color_palette()[2], alpha=0.25)
    plt.scatter(X_train, Y_train)



