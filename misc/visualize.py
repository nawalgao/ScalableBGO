
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk", font_scale=1.4)
import sys
sys.path.append('../')


def visualize_utility1D(X_train, Y_train,
                        X_new, Y_new,
                        Xgrid, Mgrid, Stdgrid):
    """
    Visualize 1D utility funciton values
    Xgrid : grid states
    Mgrid : mean of GP at those finite grid points
    Vargrid : variance of GP at those finite grid points
    """
    lower = Mgrid - 2*Stdgrid
    upper = Mgrid + 2*Stdgrid

    #plt.figure(figsize=(12,8))
    plt.scatter(X_train, Y_train, marker = 'x', color = 'r')
    plt.scatter(X_new, Y_new, marker = 'o', color = 'g')
    
    line, = plt.plot(Xgrid, Mgrid, lw = 2)
    plt.fill_between(Xgrid[:,0], lower, upper,
                     color = line.get_color(), alpha = 0.25)
    plt.xlabel('Feature values')
    plt.ylabel('GP model values')
    plt.title('GP Predictions')
    return

def visualize_parallel_evals(X_train, Y_train, X_pending, Y_pending,
                             X_new, Y_new, Xgrid, Mgrid, Stdgrid):
    """
    Visualize 1D utility funciton values
    Xgrid : grid states
    Mgrid : mean of GP at those finite grid points
    Vargrid : variance of GP at those finite grid points
    """
    lower = Mgrid - 2*Stdgrid
    upper = Mgrid + 2*Stdgrid

    #plt.figure(figsize=(12,8))
    plt.scatter(X_train, Y_train, marker = 'o', color = 'r')
    plt.scatter(X_new, Y_new, marker = 'x', color = 'g')
    plt.scatter(X_pending, Y_pending, marker = 'x', color = 'b')
    line, = plt.plot(Xgrid, Mgrid, lw = 2)
    plt.fill_between(Xgrid[:,0], lower, upper,
                     color = line.get_color(), alpha = 0.25)
    plt.xlabel('Feature values')
    plt.ylabel('GP model values')
    plt.title('GP Predictions')
    return
    


def diff_ut(X_train, Y_train,
            X_new, Y_new,
            Xgrid, meanmat, varmat, num_gps = 5):
    """
    Different utilities along with the associated uncertainities
    """

    for i in range(num_gps):
        visualize_utility1D(X_train, Y_train,
                            X_new, Y_new,
                            Xgrid, meanmat[i,:],varmat[i,:])
#        if savefig:
#            plt.savefig(('../figs/gp_diff_hyper/T' + 
#                         str(trial_num) + 'I' + str(i) + '.png'), dpi = 600)
    return