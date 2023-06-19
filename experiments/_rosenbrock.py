import torch
from functorch import vmap
from SOBER._prior import MixedCategoricalPrior
from .funcs._synthetic_function import RosenbrockFunction

def setup_rosenbrock():
    """
    Set up the experiments with Rosenbrock function
    
    Return:
    - prior: class, the function of mixed prior
    - TestFunction: class, the function that returns true Ackley function value
    """
    n_dims_cont = 1 # number of dimensions for continuous variables
    n_dims_disc = 6 # number of dimensions for categorical variables
    n_discrete = 4  # number of categories for categorical variables
    n_dims = n_dims_cont + n_dims_disc # total number of dimensions
    _min, _max = -4, 11 # the lower and upper bound of continous varibales
    
    prior = MixedCategoricalPrior(n_dims_cont, n_dims_disc, n_discrete, _min, _max)
    TrueFunction = RosenbrockFunction
    TestFunction = vmap(RosenbrockFunction)
    
    return prior, TestFunction