import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.models.transforms import Standardize


def generate_random_gp(dim: int, num_train: int, standardized: bool = True):
    r"""
    Returns a fitted gp trained on random input. Useful for testing purposes.

    Args:
        dim: Input dimension
        num_train: Number of training points
        standardized: If True, the train outcomes are standardized

    Returns:
        A fitted SingleTaskGP model
    """
    if standardized:
        transform = Standardize(m=1)
    else:
        transform = None
    train_X = torch.rand(num_train, dim)
    train_Y = torch.rand(num_train, 1)
    model = SingleTaskGP(train_X, train_Y, outcome_transform=transform)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model
