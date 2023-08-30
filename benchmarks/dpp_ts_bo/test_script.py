"""Test script to verify that implementation is functioning."""
import torch
import gpytorch

from gp import GP
from snippet_dppts import MCMC_DPP_Batched_TS_GP

# Set up a random function.
X = torch.randn(10, 2)
def F(x):
    torch.manual_seed(0)
    y = torch.randn(x.shape[0])
    return y
y = F(X)

# Set up a GP
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gpytorch_model = ExactGPModel(X, y, likelihood)

wrapped_model = GP(gpytorch_model)
gp = MCMC_DPP_Batched_TS_GP(X, F, wrapped_model)
X_cand = torch.randn(50, 2)
output = gp.step(X_cand, batch_size=3)
batch = output['x_batch']
print(f'Batch: {batch}')
