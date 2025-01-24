"""Extends GPyTorch GP for use with snippets."""
from typing import Tuple
import torch
import gpytorch
from botorch.fit import fit_gpytorch_mll

class GP():
    """GP class with additional methods."""
    def __init__(self, gp) -> None:
        self.GP = gp
        self.d = self.GP.train_inputs[0].size(1)

    @property
    def s(self):
        return self.GP.likelihood.noise.sqrt()

    """
    def fit_gp(self, x: torch.Tensor, y: torch.Tensor, iterative: bool = False):
        # Fit the GP hyperparameters.
        # Set to Scipy defaults.
        model = self.GP
        model.set_train_data(x, y, strict=False)
        optimiser = torch.optim.LBFGS(
            [{'params': model.parameters()}, ],
            lr=1,
            max_iter=5000,
            max_eval=5000,
            history_size=10,
            line_search_fn='strong_wolfe'
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        def closure():
            optimiser.zero_grad()
            loss = -mll(model.likelihood(model(x)), y)
            loss.backward()
            return loss
        optimiser.step(closure)
        return None
    """
    
    def fit_gp(self, x: torch.Tensor, y: torch.Tensor, iterative: bool = False):
        """Fit the GP hyperparameters."""
        # Set to Scipy defaults.
        model = self.GP
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return None

    def mean_var(
        self, x: torch.Tensor, full: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean and variance.
        
        :param x: Input, shape [N, D]
        :param full: Whether to return full covariance.
        :return: Predictive mean and variance.
        """
        model = self.GP
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model(x)
        mean = posterior.mean
        var = posterior.covariance_matrix
        if not full:
            var = var.diag()
        return mean, var

    def sample_and_max(self, xtest: torch.Tensor):
        # Sample from GP at xtest
        model = self.GP
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model(xtest) 
        samples = posterior.sample()
        max_ind = torch.argmax(samples)
        return xtest[max_ind], samples[max_ind]
