import time
import torch
import warnings
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from experiments._maxsat import setup_maxsat
from SOBER._sober import Sober
warnings.filterwarnings('ignore')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_rbf_model(X, Y):
    """
    Set up the Gaussian process model with RBF kernel.
    
    Args:
    - X: torch.tensor, the observed input X
    - Y: torch.tensor, the observed outcome Y
    
    Return:
    - model: gpytorch.models, function of GP model.
    """
    base_kernel = RBFKernel()
    covar_module = ScaleKernel(base_kernel)

    # Fit a GP model
    train_Y = (Y - Y.mean()) / Y.std()
    train_Y = train_Y.view(-1).unsqueeze(1)
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    model = SingleTaskGP(X, train_Y, likelihood=likelihood, covar_module=covar_module)
    return model

def fit_model(X, Y):
    """
    Optimise the hyperparameters of Gaussian process model using L-BFGS-B (BoTorch optimizer)
    
    Args:
    - X: torch.tensor, the observed input X
    - Y: torch.tensor, the observed outcome Y
    
    Return:
    - model: gpytorch.models, the optimised GP model.
    """
    model = set_rbf_model(X, Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


if __name__ == "__main__":
    seed=0
    torch.manual_seed(seed)  # random seed
    
    # set up the experiments
    prior, TrueFunction = setup_maxsat()
    
    batch_size = 200   # number of batch samples
    n_rec = 20000      # number of candidates sampled from pi
    n_nys = 500        # number of samples for Nyström approximation
    n_init = 100       # number of initial samples
    n_iterations = 15  # number of iterations (batches)

    # initial sampling
    Xall = prior.sample(n_init)
    Yall = TrueFunction(Xall)
    model = set_rbf_model(Xall, Yall)
    sober = Sober(prior, model)

    results = []
    obj = None
    for n_iter in range(n_iterations):
        start = time.monotonic()
        model = fit_model(Xall, Yall)
        sober.update_model(model)
        X = sober.next_batch(
            n_rec,
            n_nys,
            batch_size,
            calc_obj=None,  # Whether using an acquisition function
            verbose=False,   # Whether showing the detailed progress
        )
        end = time.monotonic()
        interval = end - start

        Y = TrueFunction(X)
        Xall = torch.cat((Xall, X), dim=0)
        Yall = torch.cat((Yall, Y), dim=0)

        print(f"{len(Xall)}) Best value: {Yall.max().item():.5e}")
        print(f"Acquisition time [s]: {interval:.5e}, per sample [ms]: {interval/batch_size*1e3:.5e}")
        #results.append([interval, Yall.max().item()])
