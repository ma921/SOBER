import torch
from SOBER._prior import Uniform
from ._synthetic_function import HartmannFunction

def setup_hartmann():
    """
    Set up the experiments with Hartmann function
    
    Return:
    - prior: class, the function of mixed prior
    - TestFunction: class, the function that returns true Ackley function value
    """
    n_dims_cont = 6      # number of dimensions for continuous variables
    n_dims = n_dims_cont # total number of dimensions
    _min, _max = 0, 1    # the lower and upper bound of continous varibales
    
    # Set up the bounds of the continuous domain
    mins = _min * torch.ones(n_dims_cont)
    maxs = _max * torch.ones(n_dims_cont)
    bounds = torch.vstack([mins, maxs])
    
    prior = Uniform(bounds)
    TestFunction = HartmannFunction
    
    return prior, TestFunction