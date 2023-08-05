import torch
import gpytorch
import numpy as np
import pandas as pd
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel
from gpytorch.distributions import MultivariateNormal


def default_postprocess_script(x):
    return x

def batch_tanimoto_sim(
    x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    # Tanimoto distance is proportional to (<x, y>) / (||x||^2 + ||y||^2 - <x, y>) where x and y are bit vectors
    assert x1.ndim >= 2 and x2.ndim >= 2
    dot_prod = torch.matmul(x1, torch.transpose(x2, -1, -2))
    x1_sum = torch.sum(x1**2, dim=-1, keepdims=True)
    x2_sum = torch.sum(x2**2, dim=-1, keepdims=True)
    return (dot_prod + eps) / (
        eps + x1_sum + torch.transpose(x2_sum, -1, -2) - dot_prod
    )


class BitDistance(torch.nn.Module):
    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def _sim(self, x1, x2, postprocess, x1_eq_x2=False, metric="tanimoto"):
        # Branch for Tanimoto metric
        if metric == "tanimoto":
            res = batch_tanimoto_sim(x1, x2)
            res.clamp_min_(0)  # zero out negative values
            return self._postprocess(res) if postprocess else res
        else:
            raise RuntimeError(
                "Similarity metric not supported. Available options are 'tanimoto'"
            )


class BitKernel(gpytorch.kernels.Kernel):
    def __init__(self, metric="", **kwargs):
        super().__init__(**kwargs)
        self.metric = metric

    def forward(self, x1, x2, **params):
        return self.covar_dist(x1, x2, **params)

    def covar_dist(
        self,
        x1,
        x2,
        last_dim_is_batch=False,
        dist_postprocess_func=default_postprocess_script,
        postprocess=True,
        **params,
    ):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        # torch scripts expect tensors
        postprocess = torch.tensor(postprocess)

        res = None

        # Cache the Distance object or else JIT will recompile every time
        if (
            not self.distance_module
            or self.distance_module._postprocess != dist_postprocess_func
        ):
            self.distance_module = BitDistance(dist_postprocess_func)

        res = self.distance_module._sim(
            x1, x2, postprocess, x1_eq_x2, self.metric
        )

        return res

class TanimotoKernel(BitKernel):
    is_stationary = False
    has_lengthscale = False

    def __init__(self, **kwargs):
        super(TanimotoKernel, self).__init__(**kwargs)
        self.metric = "tanimoto"

    def forward(self, x1, x2, diag=False, **params):
        if diag:
            assert x1.size() == x2.size() and torch.equal(x1, x2)
            return torch.ones(
                *x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device
            )
        else:
            return self.covar_dist(x1, x2, **params)
        
class TanimotoGP(SingleTaskGP):
    def __init__(self, train_X, train_Y):
        super().__init__(train_X, train_Y, GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
