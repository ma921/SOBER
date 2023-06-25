import torch
from functorch import vmap
from SOBER._prior import MixedCategoricalPrior
from ._synthetic_function import RosenbrockFunction

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
    
    # Set up the bounds of the continuous domain
    mins = _min * torch.ones(n_dims_cont)
    maxs = _max * torch.ones(n_dims_cont)
    bounds = torch.vstack([mins, maxs])
    
    # Set up the categories of discrete variables
    categories = torch.tensor([
        [-4,  1,  6, 11],
    ]).float().repeat(n_dims_disc,1)
    
    prior = MixedCategoricalPrior(
        n_dims_cont, 
        n_dims_disc, 
        categories, 
        bounds, 
        continous_first=True,
    )
    TrueFunction = RosenbrockFunction
    TestFunction = vmap(RosenbrockFunction)
    
    return prior, TestFunction