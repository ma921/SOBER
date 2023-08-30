import warnings

import torch
from botorch import gen_candidates_scipy
from torch import Tensor
from botorch.utils import draw_sobol_samples, batched_multinomial
from .base import CompositeSampler
from botorch.models import SingleTaskGP
from typing import Optional, Union


def decoupled_ts(posterior_sampler: CompositeSampler, num_draws: int, d: int) -> Tensor:
    r"""
    Apply TS algorithm on a discretization of the space using num_draws samples.
    Uses samples from decoupled sampler.
    Note: This assumes that the solution space is [0, 1]^d.

    Args:
        posterior_sampler: A decoupled sampler object for sampling from the posterior.
        num_draws: Number of points used to discretize the space - a power of 2
            recommended
        d: dimension of the solution space

    Returns:
        `sample_shape x d` tensor of maximizers
    """
    X = draw_sobol_samples(
        torch.tensor([[0.0], [1.0]]).repeat(1, d), n=num_draws, q=1
    ).squeeze(-2)
    with torch.no_grad():
        samples = posterior_sampler(X).squeeze(-1)
    arg_max = torch.argmax(samples, dim=-1)
    return X[arg_max]


def exact_ts(
    model: SingleTaskGP,
    num_draws: int,
    d: int,
    q: int = 1,
    device: Optional[Union[torch.device, str]] = None,
) -> Tensor:
    r"""
    Apply TS algorithm on a discretization of the space using num_draws samples.
    Uses samples from the GP posterior.
    Note: This assumes that the solution space is [0, 1]^d.

    Args:
        model: A fitted GP model
        num_draws: Number of points used to discretize the space - a power of 2
            recommended
        d: dimension of the solution space
        q: number of parallel samples
        device: Option to specify cpu / gpu. If None, defaults to GPU if available.
        # TODO: why not infer this from the model?

    Returns:
        `q x d` tensor of maximizers
    """
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        device = torch.device(device) if isinstance(device, str) else device
    X = (
        draw_sobol_samples(torch.tensor([[0.0], [1.0]]).repeat(1, d), n=num_draws, q=1)
        .squeeze(-2)
        .to(device)
    )
    with torch.no_grad():
        samples = (
            model.posterior(X).rsample(sample_shape=torch.Size([q])).squeeze(-1).t()
        )
    arg_max = torch.argmax(samples, dim=-1)
    return X[arg_max]


def continuous_decoupled_ts(
    posterior_sampler: CompositeSampler,
    num_restarts: int,
    raw_samples: int,
    d=int,
) -> Tensor:
    r"""
    Apply TS algorithm on a sample function drawn using decoupled sampler. The sample
    function is optimized via LBFGS to obtain the thompson sample.
    Note: This assumes that the solution space is [0, 1]^d.

    Args:
        posterior_sampler: A decoupled sampler object for sampling from the posterior.
        num_restarts: Number of restart points used for optimizing the sample paths.
        raw_samples: Number of raw samples to use for picking the restart points.
        d: dimension of the solution space

    Returns:
        `sample_shape x d` tensor of maximizers
    # TODO: test with CUDA
    """
    if raw_samples < num_restarts:
        warnings.warn(
            "raw samples is less than num_restarts. Using "
            f"raw_samples={num_restarts} instead.",
            RuntimeWarning,
        )
        raw_samples = num_restarts
    sample_shape = posterior_sampler.sample_shape
    if len(sample_shape) > 1:
        warnings.warn(
            "Continuous TS was not tested with multi-dim sample_shape."
            f"Use at your own risk! Sample_shape: {sample_shape}",
            RuntimeWarning,
        )

    # generate the restart points via the softmax heuristic
    raw_X = draw_sobol_samples(
        torch.tensor([[0.0], [1.0]]).repeat(1, d),
        n=raw_samples,
        q=1,
        batch_shape=sample_shape,
    ).squeeze(-2).permute(*range(1, len(sample_shape) + 1), 0, -1)
    raw_Y = posterior_sampler(raw_X)
    Y_std = raw_Y.std(dim=-2, keepdim=True)
    max_val, max_idx = torch.max(raw_Y, dim=-2)
    Z = (raw_Y - raw_Y.mean(dim=-2, keepdim=True)) / Y_std
    weights = torch.exp(Z)
    while torch.isinf(weights).any():
        Z *= 0.5
        weights = torch.exp(Z)
    idcs = batched_multinomial(
        weights=weights.squeeze(-1),
        num_samples=num_restarts,
    )
    # This will lead to some duplicates but is probably
    # better than checking in a for loop
    idcs[..., -1:] = max_idx
    ics = raw_X.gather(
        dim=-2,
        index=idcs.unsqueeze(-1).expand(*[-1] * idcs.dim(), d),
    )

    # optimize from these restart points
    sol, val = gen_candidates_scipy(
        initial_conditions=ics,
        acquisition_function=posterior_sampler,
        lower_bounds=0.0,
        upper_bounds=1.0,
    )
    argmax = val.argmax(dim=-2)
    return torch.gather(
        sol,
        dim=-2,
        index=argmax.unsqueeze(-1).expand(*[-1] * (argmax.dim()), d),
    ).squeeze(-2)
