#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 12:28:02 2018

@author: nimishawalgaonkar
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../')
import time

from learning_objective import gaussian_process
from optimize import optimize
from misc import visualize

# Need to check for time it takes to minimize a given objective function
# Need to find time it takes to select new point based on training and pending evaluations

# distance from optimum vs. total number of iterations
# distance from optimum vs. total number of function evals


# Example details
# X_train_initial : 2 states (s1, s2)
# X_pending : 2 states (s3, s4)
# New points select : 2 states (s5, s6)

num_evals = 10
cap = 2
states_evaluated = 2
obj_func = 'gp_test'
X_train = np.linspace(-1, 1, 2)[:,None]
Y_train = gaussian_process.gaussian_process(X_train)[:,0]
    
X_test = np.linspace(np.min(X_train), np.max(X_train), 100)[:,None]
Y_test = gaussian_process.gaussian_process(X_test)[:,0]

optimum_state = X_test[np.argmax(Y_test)]

#------------------------------------------------------------------------------
# 1.
# GP-MLE + Snoek Fantasy EI
trial_num = 6
num_fantasies = 8
Opt = optimize.TestParallelSnoekGPMLE()
Opt.init_pending(cap = 2)

tt = []
avg_dist = []
for i in range(num_evals):
    
    print('Iter num:')
    print(i)
    
    new_states = np.zeros((cap, Opt.X_pending.shape[1]))
    start_time = time.time()
    for j in range(cap):
        Opt.gen_fantasies(num_fantasies)
        _, _, X_suggest = Opt.suggest()
        Opt.X_pending = np.vstack([Opt.X_pending, X_suggest])
        new_states[j,:] = X_suggest
    time_tk = time.time() - start_time
    tt.append(time_tk)
    xpendtotrain = Opt.X_pending[:states_evaluated, :]
    Opt.X_pending = Opt.X_pending[states_evaluated:, :]
    plt.figure()
    GPO = Opt.ModelconditionTrain
    mean, sd = GPO.gp_mle_test()
    #mean = Opt.mean
    #sd = Opt.sd
    print ('X_train')
    print (Opt.X_train)
    print ('X_pending')
    print (xpendtotrain)
    print ('X_suggest')
    print (Opt.X_pending)
    
    avg_dist_suggest_optimum = np.linalg.norm(Opt.X_pending - optimum_state)
    avg_dist.append(avg_dist_suggest_optimum)
    visualize.visualize_parallel_evals(Opt.X_train,
                                       Opt.Y_train,
                                       xpendtotrain,
                                       gaussian_process.gaussian_process(xpendtotrain)[:,0],
                                       Opt.X_pending,
                                       gaussian_process.gaussian_process(Opt.X_pending)[:,0],
                                       Opt.X_test, mean, sd)
    plt.savefig(('../figs/models/model' +
                 str(trial_num) + '/iter' + str(i) + '_f_' + str(num_fantasies) + '.png'),
                dpi = 600)
    
    
    Opt.X_train = np.vstack([Opt.X_train, xpendtotrain])
    ypendtotrain = gaussian_process.gaussian_process(xpendtotrain)[:,0]
    Opt.Y_train = np.append(Opt.Y_train, ypendtotrain)
    
    np.savetxt('../results/time/SnoekTime' + str(num_fantasies) + '.txt', np.array(tt))
    np.savetxt('../results/time/SnoekDist' + str(num_fantasies) + '.txt', np.array(avg_dist))

#------------------------------------------------------------------------------
## 2.
## GP-MLE + Constant Liar Strategy 
#
#trial_num = 7
#
#Opt = optimize.ConstantLiarEI()
#Opt.init_pending(cap = 2)
#
#tt = []
#avg_dist = []
#for i in range(num_evals):
#    
#    print('Iter num:')
#    print(i)
#    
#    new_states = np.zeros((cap, Opt.X_pending.shape[1]))
#    start_time = time.time()
#    for j in range(cap):
#        _, _, X_suggest = Opt.suggest()
#        Opt.X_pending = np.vstack([Opt.X_pending, X_suggest])
#        new_states[j,:] = X_suggest
#    time_tk = time.time() - start_time
#    tt.append(time_tk)
#    xpendtotrain = Opt.X_pending[:states_evaluated, :]
#    Opt.X_pending = Opt.X_pending[states_evaluated:, :]
#    plt.figure()
#    Op = optimize.Optimize(Opt.X_train, Opt.Y_train, Opt.X_test)
#    Op.gp_mle_train(False, False)
#    mean, sd = Op.gp_mle_test()
#    #mean = Opt.mean
#    #sd = Opt.sd
#    print ('X_train')
#    print (Opt.X_train)
#    print ('X_pending')
#    print (xpendtotrain)
#    print ('X_suggest')
#    print (Opt.X_pending)
    
#    avg_dist_suggest_optimum = np.linalg.norm(Opt.X_pending - optimum_state)
#    avg_dist.append(avg_dist_suggest_optimum)
#    visualize.visualize_parallel_evals(Opt.X_train,
#                                       Opt.Y_train,
#                                       xpendtotrain,
#                                       gaussian_process.gaussian_process(xpendtotrain)[:,0],
#                                       Opt.X_pending,
#                                       gaussian_process.gaussian_process(Opt.X_pending)[:,0],
#                                       Opt.X_test, mean, sd)
#    plt.savefig(('../figs/models/model' +
#                 str(trial_num) + '/iter' + str(i) + '.png'),
#                dpi = 600)
#    
#    
#    Opt.X_train = np.vstack([Opt.X_train, xpendtotrain])
#    ypendtotrain = gaussian_process.gaussian_process(xpendtotrain)[:,0]
#    Opt.Y_train = np.append(Opt.Y_train, ypendtotrain)
#    
#    np.savetxt('../results/time/LiarTime.txt', np.array(tt))
#    np.savetxt('../results/time/LiarDist.txt', np.array(avg_dist))
#
#

#------------------------------------------------------------------------------
# 3.
## GP with parallel integrated Shu EI (integrated = True) - what EI I have right now.

#trial_num = 4
#
#OptimizeGP = optimize.design_of_experiments(X_train, Y_train, X_test, optimum_state,
#                                            obj_func, aq = 'multi_select_ei',
#                                            model = 'gp',
#                                            num_iter = num_evals,
#                                            cap = cap, trial = trial_num)

#------------------------------------------------------------------------------
# 4. 
## Neural Networks with last layer Bayesian

OptimizeGP = optimize.design_of_experiments(X_train, Y_train, X_test, optimum_state,
                                            obj_func, aq = 'multi_select_ei',
                                            model = 'blnn',
                                            num_iter = num_evals,
                                            cap = cap, trial = trial_num)
