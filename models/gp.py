#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:28:25 2018

@author: nimishawalgaonkar
"""

import numpy as np
import gpflow 
from gpflow.test_util import notebook_niter
import sys
sys.path.append('../')
from misc import normalization, visualize
from learning_objective import gaussian_process


class GP(object):
    def __init__(self, X, y, domain_features, normalize_input = True, normalize_output = False):
        """
        Gaussian processes
        MLE
        MCMC
        """
        self.X = X
        self.y = y
        self.domain_features = domain_features
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        if self.normalize_input:
            self.X, self.xnorm_mean, self.xnorm_sd  = normalization.zero_mean_unit_var_normalization(self.X)
            (self.domain_features,
             self.x_test_mean, self.x_test_sd) = normalization.zero_mean_unit_var_normalization(self.domain_features)
        if self.normalize_output:
            self.y, self.ynorm_mean, self.ynorm_sd = normalization.zero_mean_unit_var_normalization(self.y)
    
    def train_mle(self):
        """
        Train MLE
        """
        k = gpflow.kernels.Matern52(self.X.shape[1], lengthscales=0.3)
        m = gpflow.models.GPR(self.X, self.y[:,None], kern=k)
        m.likelihood.variance = 0.01
        gpflow.train.ScipyOptimizer().minimize(m)
        self.m = m
        
    def test_mle(self, xtest = None):
        """
        Test MLE
        """
        if xtest is None:
            
            mean, var = self.m.predict_f(self.domain_features)
        else:
            xtt, xtnorm_mean, xtnorm_sd  = normalization.zero_mean_unit_var_normalization(xtest)
            mean, var = self.m.predict_f(xtest)
        mean = mean[:,0]
        var = var[:,0]
        sd = np.sqrt(var)
    
        return mean, sd 
    
    def train_mcmc(self, num_samples = 1000, burn = 500, thin = 2,
                        epsilon = 0.05, lmin = 10, lmax = 20):
        """
        MCMC
        """
        k = gpflow.kernels.Matern52(input_dim = self.X.shape[1], ARD=True)
        l = gpflow.likelihoods.Gaussian()
        m = gpflow.models.GPMC(self.X, self.y[:,None], k, l)
        m.clear()
        m.likelihood.variance.prior = gpflow.priors.Gamma(1., 1.)
        m.kern.lengthscales.prior = gpflow.priors.Gamma(1., 1.)
        m.kern.variance.prior = gpflow.priors.Gamma(1.,1.)
        m.compile()
        #o = gpflow.train.AdamOptimizer(0.01)
        #o.minimize(m, maxiter=notebook_niter(15)) # start near MAP
        gpflow.train.ScipyOptimizer().minimize(m)
        sampler = gpflow.train.HMC()
        samples = sampler.sample(m, num_samples= notebook_niter(num_samples), burn = notebook_niter(burn),
                                 thin = notebook_niter(thin), epsilon= epsilon,
                                 lmin= lmin, lmax= lmax, logprobs=False)
        self.m = m
        self.samples = samples
        print('GP MCMC training is done ...')
        
        
    def test_mcmc(self):
        """
        MCMC
        """
        # Posterior function values for normalized state matrix X_norm
        meanmat = np.zeros(shape = (self.samples.shape[0], self.domain_features.shape[0]))
        varmat = np.zeros(shape = (self.samples.shape[0], self.domain_features.shape[0]))
        for i, s in self.samples.iterrows():
            print('GP samples for posterior hyper parameter sample :')
            print(i)
    
            self.m.assign(s)
            mean, var = self.m.predict_f(self.domain_features)
            meanmat[i,:] = mean[:,0]
            varmat[i,:] = var[:,0]
         
        sdmat = np.sqrt(varmat)
        
        return meanmat, sdmat
    
    def post_samples(self, x, gp_samples):
        """
        MCMC
        """
        # Posterior function values for normalized state matrix X_norm
        #allgps = np.array([])
        allgps = np.zeros((gp_samples, x.shape[0]))
        for i, s in self.samples.iterrows():
            print('GP samples for posterior hyper parameter sample :')
            print(i)
            self.m.assign(s)
            post_samples = self.m.predict_f_samples(x, 1)[:,:,0]
            allgps[i,:] = post_samples
            if allgps.shape[0] > gp_samples:
                print ('Got the required GP samples')
                break
        
        return allgps
    
    def gp_mle_samples(self, x, num):
        post_samples = self.m.predict_f_samples(x, num)[:,:,0]
        
        return post_samples
        
        
            
        
        
        
        
    
    
if __name__ == '__main__':
    
    
    X_train = np.linspace(-1, 1, 2)[:,None]
    Y_train = gaussian_process.gaussian_process(X_train)[:,0]
    X_test = np.linspace(np.min(X_train), np.max(X_train), 200)[:,None]
    
    # Train GP with HMC sampling
    GPO2 = GP(X_train, Y_train, X_train, False, False)
    
    # Assess the model fit
    GPO2.train_mcmc()
    #m, var = GPO2.test_mcmc()
    
#    # Visualize the model fit
#    visualize.diff_ut(X_train, Y_train, X_test, meanmat, varmat, trial_num = 1, 
#                      savefig = True, num_gps = 10)
#    
    
#    plt.scatter(X_train, Y_train, color = 'g')
#    plt.scatter(X_train, meanmat[0,:], color = 'r')
        
        
        
