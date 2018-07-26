"""
@Author: Rui Shu
@Date: 4/11/15

Provides a proxy hidden function for running of optimizer and mpi_optimizer
"""

import numpy as np
import time
from gaussian_mix import gaussian_mix as gm
from hartmann import hartmann as hm
from gaussian_process import gaussian_process as gp

# Definitions for which function to evaluate
HM = 0     # hartmann
GP = 1     # gaussian process realization
GM = 2     # gaussian mixture

# Set it
method = 1 # currently set as GP

def get_settings(lim_domain_only=False):
    """ Get settings for the optimizer.
    """
    # Settings
    if method == HM:
        lim_domain = np.array([[0., 0., 0., 0.],
                               [ 1.,  1., 1., 1.]])
    elif method == GM:
        lim_domain = np.array([[-1., -1.],
                               [ 1.,  1.]])
    elif method == GP:
        lim_domain = np.array([[-1.],
                               [ 1.]])

    if lim_domain_only:
        return lim_domain

    init_size = 2
    additional_query_size = 30
    selection_size = 1

    # Get initial set of locations to query
    init_query = np.random.uniform(0, 1, size=(init_size, lim_domain.shape[1]))

    # Establish the grid size to use. 
    if method == HM:
        r = np.linspace(-1, 1, 15)
        X = np.meshgrid(r, r, r, r)


















































