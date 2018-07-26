#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:48:15 2018

@author: nimishawalgaonkar
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import gpflow
import time
sns.set_context('talk')
sns.set_style('white')
import sys
sys.path.append('../')
sys.path.append('../models/')
from models import ll_bayesian_nn, gp
from learning_objective import gaussian_process
from misc import normalization, visualize



class Optimize(object):
    def __init__(self, X, y, domain_features):
        
        self.X = X
        self.y = y
        
        self.domain_features = domain_features
        
    
    def blnn_train_eval(self, num_epochs = 1000):
        """
        Last Layer Bayesian Neural Network
        """
        
        # Fit the required model
        LLBNN = ll_bayesian_nn.LLBayesianNN(self.X, self.y,
                                            batch_size = self.X.shape[0],
                                            num_epochs = num_epochs)
        LLBNN.train()
        print('.'*40)
        print('Training is done')
        print('.'*40)
        
        # Predict the mean and variance of predictions at the domain points
        # using last layer bayesian neural network
        print('.'*40)
        print ('Using surrogate LLBNN, predicting for the domain features')
        print('.'*40)
        
        self.mean, self.sd = LLBNN.test(self.domain_features)
    
    
    def gp_mle_train(self, normalize_input = True, normalize_output = True):
        self.GPO = gp.GP(self.X, self.y, self.domain_features,
                        normalize_input, normalize_output)
        self.GPO.train_mle()
    
    def gp_mle_test(self, xtest = None):
        """
        MLE GP test
        """
        self.mean, self.sd = self.GPO.test_mle(xtest)
        
        return self.mean, self.sd
    
    def gp_mle_sample(self, x, num_samples):
        samples = self.GPO.gp_mle_samples(x, num_samples)
        return samples
          
    def gp_mcmc_train(self, normalize_input = True, normalize_output = True):
        """
        MCMC GP train
        """
        self.GPO = gp.GP(self.X, self.y, self.domain_features,
                        normalize_input, normalize_output)
        
        
        self.GPO.train_mcmc(num_samples = 10, burn = 500,
                   thin = 2, epsilon = 0.05,
                   lmin = 10, lmax = 20)
    
    def gp_mcmc_test(self, xtest = None):
        """
        MCMC GP test
        """
        
        self.meanmat, self.sdmat = self.GPO.test_mcmc(xtest)
        
        return self.meanmat, self.sdmat
    
    def gp_mcmc_sample(self, x, samples):
        """
        GP samples at x (for different hyperparameters)
        Total number of samples = hyper_parameter samples* gp_samples_for_each
        """
        allgps = self.GPO.post_samples(x, samples)
        
        return allgps
    
    def EI(self):
        max_obj = np.max(self.y)
        diff = self.mean - max_obj
        Z = diff/self.sd
        pdf = scipy.stats.norm.pdf(Z)
        cdf = scipy.stats.norm.cdf(Z)
        EI = diff*cdf + self.sd*pdf
        
        #sd = 0 indices
        sd_zero_ind = np.where(self.sd == 0)[0]
        EI[sd_zero_ind] = 0
        #return EI
        
        max_EI = np.max(EI)
        max_EI_ind = np.argmax(EI)
        next_state = self.domain_features[max_EI_ind, :]
        
        return EI, max_EI, next_state
    
    def integratedEI(self):
        """
        Integrated expected improvement. Can only be used with `GPflow MCMC` instance.
        
        Parameters
        ----------
        tau: float
            Best observed function evaluation
        meanmcmc: array-like
            Means of posterior predictive distributions after sampling.
        stdmcmc
            Standard deviations of posterior predictive distributions after sampling.
            
        Returns
        -------
        float:
            Integrated Expected Improvement
        """
        max_obj = np.max(self.y)
        mdiff = self.meanmat - max_obj
        Z = mdiff/self.sdmat
        pdf = scipy.stats.norm.pdf(Z)
        cdf = scipy.stats.norm.cdf(Z)
        exp_imp = mdiff*cdf + self.sdmat*pdf
        mean_exp_imp = np.mean(exp_imp, axis = 0)
        
        mean_max_ind = np.argmax(mean_exp_imp)
        max_exp_imp = mean_exp_imp[mean_max_ind]
        next_state = self.domain_features[mean_max_ind.astype(int),:]
        
        return mean_exp_imp, max_exp_imp, next_state
   

    def check_point(self, selected_index, order):
        z_score = abs(self.mean[order] - self.mean[selected_index])/self.sd[selected_index]
        return (stats.norm.cdf(-z_score)*2) < 0.5
    
    def integrated_check_point(self, selected_index, order):
        z_scoremat = abs(self.meanmat[:,order] - self.meanmat[:,selected_index])/self.sdmat[:,selected_index]
        cd = (stats.norm.cdf(-z_scoremat)*2) < 0.5
        keep = np.sum(cd)/cd.shape[0]
        return keep
        
    def parallelEI(self, integrated = False, cap = 5):
        
        if integrated:
            ei, _, _ = self.integratedEI()
            if np.max(ei) <= 0:
                # If no good points, do pure exploration
                sdmean = np.mean(self.sdmat, axis = 1)
                sig_order = np.argsort(-1*sdmean, axis=0)
                select_indices = sig_order[:cap].tolist()
            else:
                ei_order = np.argsort(-1*ei)
                select_indices = [ei_order[0]]
                for candidate in ei_order[:]:
                    keep = True
                    for selected_index in select_indices:
                        keep = keep*self.integrated_check_point(selected_index, candidate)
                    if keep and (ei[candidate] > 0):
                            select_indices.append(candidate)
                    if len(select_indices) == cap: # Number of points to select
                        break
                if len(select_indices) < cap:
                    # If not enough good points, append with exploration
                    sdmean = np.mean(self.sdmat, axis = 1)
                    sig_order = np.argsort(-sdmean, axis=0)
                    add_indices = sig_order[:(cap-len(select_indices))].tolist()
                    select_indices.extend(add_indices)
        else:
            ei, _, _ = self.EI()
    
            if np.max(ei) <= 0:
                # If no good points, do pure exploration
                sig_order = np.argsort(-self.sd, axis=0)
                select_indices = sig_order[:cap].tolist()
            else:
                ei_order = np.argsort(-1*ei, axis=0)
                #print ('EI_order')
                #print (ei_order)
                select_indices = [ei_order[0]]
                for candidate in ei_order[:]:
                    keep = True
                    #print ('Candidate')
                    #print (candidate)
                    
                    for selected_index in select_indices:
                        keep = keep*self.check_point(selected_index, candidate)
                        #print ('Selected index')
                        #print (selected_index)
                    
                    #print ('ei_candidate')
                    #print (ei[candidate])
                    #print ('keep')
                    #print (keep)
                    if keep and (ei[candidate] > 0):
                            select_indices.append(candidate)
                    if len(select_indices) == cap: # Number of points to select
                        break
                if len(select_indices) < cap:
                    # If not enough good points, append with exploration
                    sig_order = np.argsort(-self.sd, axis=0)
                    add_indices = sig_order[:(cap-len(select_indices))].tolist()
                    select_indices.extend(add_indices)
        
        new_points = np.atleast_2d(self.domain_features[select_indices, :])
        
        return new_points
   

class ConstantLiarEI(object):
    """
    Constant Liar Strategy
    """
    def __init__(self):
        self.X_train = np.linspace(-1, 1, 2)[:,None]
        self.Y_train = gaussian_process.gaussian_process(self.X_train)[:,0]
        self.X_test = np.linspace(np.min(self.X_train),
                                  np.max(self.X_train), 40)[:,None]
    
    def init_pending(self, cap = 2):
        Opt = Optimize(self.X_train, self.Y_train, self.X_test)
        Opt.gp_mle_train(False, False)
        Opt.gp_mle_test()
        self.X_pending = Opt.parallelEI(integrated = False, cap = cap)
        return self.X_pending
    
    def suggest(self):
        X_train = self.X_train
        Y_train = self.Y_train
        X_test = self.X_test
        X_pending = self.X_pending
        
        Opt = Optimize(X_train, Y_train, X_test)
        Opt.gp_mle_train(False, False)
        
        train_pending_concat = np.vstack([X_train, X_pending])
        
        lie_m, lie_sd = Opt.gp_mle_test(X_pending)
        
        Y_train_lie = np.append(Y_train, lie_m)
        
        Opt = Optimize(train_pending_concat, Y_train_lie, X_test)
        Opt.gp_mle_train(normalize_input = False, normalize_output = False)
        mean, sd = Opt.gp_mle_test()
            
        ei_mean, ei_mean_max, next_state = Opt.EI()
        
        return ei_mean, ei_mean_max, next_state
        
        
class TestParallelSnoekEI(object):
    """
    class to test different aspects of Snoek's EI
    """
    
    def __init__(self):
        self.X_train = np.linspace(-1, 1, 2)[:,None]
        self.Y_train = gaussian_process.gaussian_process(self.X_train)[:,0]
        self.X_test = np.linspace(np.min(self.X_train),
                                  np.max(self.X_train), 40)[:,None]
    
    def init_pending(self, cap = 2):
        Opt = Optimize(self.X_train, self.Y_train, self.X_test)
        Opt.gp_mcmc_train(False, False)
        Opt.gp_mcmc_test()
        self.X_pending = Opt.parallelEI(integrated = True, cap = cap)
        return self.X_pending
    
    def gen_fantasies(self, num_fantasies):
        X_train = self.X_train
        Y_train = self.Y_train
        X_test = self.X_test
        X_pending = self.X_pending
         
        Opt = Optimize(X_train, Y_train, X_test)
        Opt.gp_mcmc_train(normalize_input = False, normalize_output = False)
        
        self.fantasies = Opt.gp_mcmc_sample(X_pending, num_fantasies)
        
        self.MconditionTrain = Opt
        
    def suggest(self):
        X_train = self.X_train # these points are already evaluated
        Y_train = self.Y_train # these points are already evaluated
        X_test = self.X_test  # these are grid points from which to select next
        X_pending = self.X_pending # these are the points for which evaluations are pending
        
        train_pending_concat = np.vstack([X_train, X_pending])
        ei_all = np.zeros((self.fantasies.shape[0], X_test.shape[0]))
        
        for i, f in enumerate(self.fantasies):
            Y_train_fantasy = np.append(Y_train, f)
            Opt = Optimize(train_pending_concat, Y_train_fantasy, X_test)
            Opt.gp_mcmc_train(normalize_input = False, normalize_output = False)
            mean, sd = Opt.gp_mcmc_test()
            
            ei_all[i,:], _, _ = Opt.integratedEI()
        
        ei_mean = np.mean(ei_all, axis = 0)
        ei_mean_max = np.max(ei_mean)
        ei_mean_max_ind = np.argmax(ei_mean)
        next_state = X_test[ei_mean_max_ind]
        
        return ei_mean, ei_mean_max, next_state
    
    
class TestParallelSnoekGPMLE(object):
    """
    Snoek's Fantasy EI with GP-MLE
    """
    def __init__(self):
        self.X_train = np.linspace(-1, 1, 2)[:,None]
        self.Y_train = gaussian_process.gaussian_process(self.X_train)[:,0]
        self.X_test = np.linspace(np.min(self.X_train),
                                  np.max(self.X_train), 40)[:,None]
    
    def init_pending(self, cap = 2):
        Opt = Optimize(self.X_train, self.Y_train, self.X_test)
        Opt.gp_mle_train(False, False)
        Opt.gp_mle_test()
        self.X_pending = Opt.parallelEI(integrated = False, cap = cap)
        return self.X_pending
    
    def gen_fantasies(self, num_fantasies):
        X_train = self.X_train
        Y_train = self.Y_train
        X_test = self.X_test
        X_pending = self.X_pending
        Opt = Optimize(X_train, Y_train, X_test)
        Opt.gp_mle_train(normalize_input = False, normalize_output = False)
        
        if X_pending.shape[0] == 0:
            print ('No pending evaluations ... moving on with EI to select new point')
        else:
            print ('Pending evals. Sampling fantasy outputs for these pending evals')
            self.fantasies = Opt.gp_mle_sample(X_pending, num_fantasies)
        
        self.ModelconditionTrain = Opt
        
        
    def suggest(self):
        X_train = self.X_train # these points are already evaluated
        Y_train = self.Y_train # these points are already evaluated
        X_test = self.X_test  # these are grid points from which to select next
        X_pending = self.X_pending # these are the points for which evaluations are pending
        
        if X_pending.shape[0] == 0:
            print ('TO DO')
            ei_mean, ei_mean_max, next_state = self.ModelconditionTrain.EI()
        else:
        
            train_pending_concat = np.vstack([X_train, X_pending])
            ei_all = np.zeros((self.fantasies.shape[0], X_test.shape[0]))
        
            for i, f in enumerate(self.fantasies):
                Y_train_fantasy = np.append(Y_train, f)
                Opt = Optimize(train_pending_concat, Y_train_fantasy, X_test)
                Opt.gp_mle_train(normalize_input = False, normalize_output = False)
                mean, sd = Opt.gp_mle_test()
            
                ei_all[i,:], _, _ = Opt.EI()
        
            ei_mean = np.mean(ei_all, axis = 0)
            ei_mean_max = np.max(ei_mean)
            ei_mean_max_ind = np.argmax(ei_mean)
            next_state = X_test[ei_mean_max_ind]
        
        return ei_mean, ei_mean_max, next_state
    
    def suggest_multi(self, cap = 2):
        new_states = []
        for i in range(cap):    
            _,_,next_state = self.suggest()
            new_states.append(new_states)
            pending_all = np.vstack([self.X_pending, next_state])
            self.X_pending = pending_all
        
        return new_states
    
        
        
def design_of_experiments(X_train, Y_train, X_domain, optimum_state,
                          obj_func, aq, model = 'blnn',
                          num_iter = 10, cap = 2, trial = 1):
    """
    Number of iterations to perform
    Number of points slected at each iteration
    """
    # Obj function as input - TODO
    print ('Aquisition input is :')
    print (aq)
    tt = []
    avg_dist = []
    for i in range(num_iter):
        print ('Iter:' + str(i))
        Opt = Optimize(X_train, Y_train, X_domain)
    
        if model == 'gp':
            Opt.gp_mle_train(normalize_input = False,
                              normalize_output = False)
            Opt.gp_mle_test()
        else:
            sys.exit('Model not found')
        
        start_time = time.time()
        if aq == 'seq_select_ei':
            print ('Aq : Sequential MLE based EI')
            _, _, new = Opt.EI()
            _, _, new = Opt.integratedEI()
        if aq == 'multi_select_ei': 
            print ('Aq : Parallel MLE based EI')
            integrated = False
            new = Opt.parallelEI(integrated, cap)
        else:
            sys.exit('Aquisition not found')
            
        time_tk = time.time() - start_time
        tt.append(time_tk)
        
        y_new = gaussian_process.gaussian_process(new)
        
        mean = Opt.mean
        sd = Opt.sd
        avg_dist_suggest_optimum = np.linalg.norm(new - optimum_state)
        avg_dist.append(avg_dist_suggest_optimum)
        plt.figure()
        visualize.visualize_utility1D(X_train, Y_train,
                                      new, y_new,
                                      Opt.domain_features, mean, sd)
        
        plt.savefig('../figs/models/model' + str(trial) + '/' + 'iter' + str(i), dpi = 600)
            
        X_train = np.vstack([X_train, new])
        Y_train = np.append(Y_train, y_new[:,0])
    
    np.savetxt('../results/time/ShuTime.txt', np.array(tt))
        
    return Opt, X_train, Y_train
    
        
        
    


if __name__ == '__main__':
    Opt = TestParallelSnoekGPMLE()
    Opt.init_pending(cap = 2)

    
    # Assume that we have X_train and Y_train as our initial evals
    # Assume that initial suggested points are X_pending (attribute of Opt)
    # Now, we want to select next two points (such that every time you select
    # a new point, one state in X_pending is evaluated ---- this is just for illustration)
    # This need not be the case, I am just showing it as an example
    
    num_evals = 15
    for i in range(num_evals):
        print('Iter num:')
        print(i)
        Opt.gen_fantasies()
        _, _, X_suggest = Opt.suggest()
        xpendtotrain = Opt.X_pending[0]
        X_pending_new = np.vstack([Opt.X_pending[1:,:], X_suggest])
        print ('X_pending_new')
        print (X_pending_new)
        
        plt.figure()
        GPO = Opt.ModelconditionTrain
        mean, sd = GPO.gp_mle_test()
        visualize.visualize_parallel_evals(Opt.X_train, Opt.Y_train,
                                           Opt.X_pending,
                                           gaussian_process.gaussian_process(Opt.X_pending)[:,0],
                                           X_suggest, gaussian_process.gaussian_process(X_suggest)[:,0],
                                           Opt.X_test, mean,
                                           sd)
        
        Opt.X_pending = X_pending_new
        Opt.X_train = np.vstack([Opt.X_train, xpendtotrain])
        ypendtotrain = gaussian_process.gaussian_process(xpendtotrain)[:,0]
        Opt.Y_train = np.append(Opt.Y_train, ypendtotrain)
        
        
    
    
    
