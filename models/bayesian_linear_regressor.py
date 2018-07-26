import numpy as np
import scipy
import statsmodels.api as sm
from sklearn.linear_model import ARDRegression
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
sns.set_style('white')
import sys
sys.path.append('../')
from misc import normalization

class BayesianARD(object):
    def __init__(self, X, y, normalize_output = False, intercept = True):
        """
        Initialization of the linear regressor object
        Inputs:
            X : last layer output of the neural network
            y : objective function value for the corresponding set of inputs
            intercept (true or false) : whether to include intecept or not
        """
        self.X = X
        self.y = y
        self.intercept = intercept
        self.normalize_output = normalize_output

    def train(self):
        """
        Train the linear regression model based on the observed dataset
        """
        if self.normalize_output:
            (self.y,
             self.norm_mean,
             self.norm_sd) = normalization.zero_mean_unit_var_normalization(self.y)
        if self.intercept:
            train_X = sm.add_constant(self.X)
        else:
            train_X = self.X
        Phi = train_X
        regressor = ARDRegression()
        regressor.fit(Phi, self.y)
        # Best sigma
        self.sigma = np.sqrt(1. / regressor.alpha_)
        # Best alpha
        self.alpha = regressor.lambda_

        A = np.dot(Phi.T, Phi) / self.sigma ** 2. + self.alpha * np.eye(Phi.shape[1])
        A = A + np.eye(A.shape[0])*1e-10
        L = scipy.linalg.cho_factor(A)

        self.m = scipy.linalg.cho_solve(L, np.dot(Phi.T, self.y) / self.sigma ** 2)  # The posterior mean of w
        self.S = scipy.linalg.cho_solve(L, np.eye(Phi.shape[1]))           # The posterior covariance of w

        return self.m, self.S, self.sigma, self.alpha

    def test(self, X_test):
        """
        Use the trained regression parameters to estimate the objective function mean and variance
        at the new point
        Inputs:
            X_test : Design matrix at new testing points
        """
        if self.intercept:
            X_test = sm.add_constant(X_test)
        Phi_p = X_test
        Y_p = np.dot(Phi_p, self.m) # The mean prediction
        V_p_ep = np.einsum('ij,jk,ik->i', Phi_p, self.S, Phi_p) # The epistemic uncertainty
        S_p_ep = np.sqrt(V_p_ep)
        #S_p_ep = S_p_ep[:,None]
        #V_p = V_p_ep + self.sigma ** 2 # Full uncertainty
        #S_p = np.sqrt(V_p)
        #S_p = S_p[:,None]
        Y_l_ep = Y_p - 2. * S_p_ep  # Lower epistemic predictive bound
        #Y_u_ep = Y_p + 2. * S_p_ep  # Upper epistemic predictive bound
        #Y_l = Y_p - 2. * S_p # Lower predictive bound
        #Y_u = Y_p + 2. * S_p # Upper predictive bound
        
        if self.normalize_output:
            Y_p_unnorm = normalization.zero_mean_unit_var_unnormalization(Y_p, self.norm_mean, self.norm_sd)
            #S_p_ep_unnorm = normalization.zero_mean_unit_var_unnormalization(S_p_ep, self.norm_mean, self.norm_sd)
            Y_l_ep_unnorm = normalization.zero_mean_unit_var_unnormalization(Y_l_ep, self.norm_mean, self.norm_sd)
            #Y_u_ep_unnorm = normalization.zero_mean_unit_var_unnormalization(Y_u_ep, self.norm_mean, self.norm_sd)
            #S_p_unnorm = normalization.zero_mean_unit_var_unnormalization(S_p, self.norm_mean, self.norm_sd)
            #Y_l_unnorm = normalization.zero_mean_unit_var_unnormalization(Y_l, self.norm_mean, self.norm_sd)
            #Y_u_unnorm = normalization.zero_mean_unit_var_unnormalization(Y_u, self.norm_mean, self.norm_sd)
        else:
            Y_p_unnorm = Y_p
            #S_p_ep_unnorm = S_p
            Y_l_ep_unnorm = Y_l_ep
            #Y_u_ep_unnorm = Y_u_ep
            #S_p_unnorm = S_p
            #Y_l_unnorm = Y_l
            #Y_u_unnorm = Y_u
        
        S_p_unnorm = (Y_p_unnorm - Y_l_ep_unnorm)/2
        
        return Y_p_unnorm, S_p_unnorm
            
        #return Y_p_unnorm, Y_l_ep_unnorm, Y_u_ep_unnorm, Y_l_unnorm, Y_u_unnorm  

if __name__ == '__main__':
    
#    X_train = np.linspace(0, 80, 2).reshape(-1, 1)
#    Y_train = 5 * X_train[:,0]
#
#    X_test = np.linspace(np.min(X_train), np.max(X_train), 200)[:,None]
#    
#    LRObj = BayesianARD(X_train, Y_train, intercept = False)
#    LRObj.train()
    
#    (Y_p, S_p_ep, Y_l_ep, Y_u_ep, S_p, Y_l, Y_u) = LRObj.test(Phi_p)

#    plt.figure()
#    plt.scatter(X, Y)
#    plt.plot(X_p, Y_p)
#    plt.fill_between(X_p.flatten(), Y_u, Y_l, color=sns.color_palette()[2], alpha=0.25)
#    plt.show(block = True)


   # Example specific required functions
   # We need a generic function that computes the design matrix

    def compute_design_matrix(X, phi):
        """
        Arguments:

        X   -  The observed inputs (1D array)
        phi -  The basis functions.
        """
        num_observations = X.shape[0]
        num_basis = phi.num_basis
        Phi = np.ndarray((num_observations, num_basis))
        for i in range(num_observations):
            Phi[i, :] = phi(X[i, :])
        return Phi

    class RadialBasisFunctions(object):
        """
        A set of linear basis functions.

        Arguments:
        X   -  The centers of the radial basis functions.
        ell -  The assumed lengthscale.
        """
        def __init__(self, X, ell):
            self.X = X
            self.ell = ell
            self.num_basis = X.shape[0]
        def __call__(self, x):
            return np.exp(-.5 * (x - self.X) ** 2 / self.ell ** 2).flatten()

    # General example to test the functions
    data = np.loadtxt('../scripts/motor.dat')
    X = data[:, 0][:, None]
    Y = data[:, 1]

    ell = 2.
    Xc = np.linspace(0, 60, 50)
    phi = RadialBasisFunctions(Xc, ell)
    Phi = compute_design_matrix(X, phi)

    X_p = np.linspace(0, 60, 100)[:, None]
    Phi_p = compute_design_matrix(X_p, phi)

    LRObj = BayesianARD(Phi, Y, normalize_output= False, intercept = False)
    LRObj.train()
    
    Y_p_unnorm, Y_l_ep_unnorm, Y_u_ep_unnorm, Y_l_unnorm, Y_u_unnorm  = LRObj.test(Phi_p)

    plt.figure()
    plt.scatter(X, Y)
    plt.plot(X_p, Y_p_unnorm)
    plt.fill_between(X_p.flatten(), Y_u_unnorm, Y_l_unnorm, color=sns.color_palette()[2], alpha=0.25)
    plt.show(block = True)
