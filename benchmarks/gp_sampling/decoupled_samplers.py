import warnings
from typing import Union, List, Iterable, Optional

import torch
from botorch.models import SingleTaskGP
from torch import Tensor
from gpytorch.lazy import delazify

from .base import CompositeSampler, BayesianLinearSampler
from .basis_functions import RandomFourierBasis, KernelBasis, Basis


def decoupled_sampler(
    model: SingleTaskGP,
    sample_shape: Union[list, tuple, torch.Size],
    num_basis: int,
    prior_basis: Basis = None,
    input_batch_shape: Optional[Iterable] = None,
) -> CompositeSampler:
    r"""
    Constructs the decoupled sampler as an instance of CompositeSampler.
    Corresponds to decoupled_sampler_gpr from the tf implementation.

    Args:
        model: The GP model
        sample_shape: The number (or batch shape) of sample paths generated.
        num_basis: Number of basis functions to use
        prior_basis: The choice of the prior basis
        input_batch_shape: When drawing samples for batches of inputs, or with a
            batched GP model, this ensures that the samples are of the appropriate shape.

    Returns:
        A CompositeSampler object

    # TODO: there's something going on when using outcome_transform=Standardize().
    """
    if getattr(model, "outcome_transform", None) is not None:
        raise RuntimeError(
            "I have observed weird behavior when using "
            "`outcome_transform=Standardize()`. Avoid outcome "
            "transforms for now."
        )
    if input_batch_shape is None:
        if model._input_batch_shape != torch.Size():
            input_batch_shape = list(model._input_batch_shape)
        else:
            input_batch_shape = list()
    else:
        input_batch_shape = list(input_batch_shape)
    if input_batch_shape != list():
        warnings.warn(
            "This has not been tested with batched inputs. Use at your own risk!",
            RuntimeWarning,
        )

    def _create_prior_fn() -> BayesianLinearSampler:
        r"""
        Creates the prior function of the decoupled sampler

        Returns:
            The corresponding BayesianLinearSampler object
        """
        basis = prior_basis
        if basis is None:
            basis = RandomFourierBasis(
                kernel=model.covar_module,
                units=num_basis,
                input_batch_shape=input_batch_shape,
            )

        def w_init(shape):
            return torch.randn(shape)

        weights = w_init(list(sample_shape) + input_batch_shape + [1, num_basis])
        return BayesianLinearSampler(
            basis=basis, weights=weights, weight_initializer=w_init
        )

    def _create_update_fn() -> BayesianLinearSampler:
        r"""
        Creates the update part of the decoupled sampler

        Returns:
            The corresponding BayesianLinearSampler object
        """
        Z = model.train_inputs[0]
        u = model.train_targets
        # add jitter here if needed
        sigma2 = model.likelihood.noise
        if model.mean_module is not None:
            u = u - model.mean_module(Z)
        if getattr(model, "outcome_transform", None) is not None:
            u = model.outcome_transform.untransform(u)[0]
        u = u.unsqueeze(-1)
        m = Z.shape[-2]
        # Kuu = delazify(model.covar_module(Z, Z))
        # Suu = Kuu + torch.eye(m).to(Kuu) * sigma2
        # Luu = torch.cholesky(Suu)
        # modified to use lazy tensor operations.
        Suu = model.covar_module(Z, Z).add_diag(sigma2)
        Luu = Suu.cholesky()
        # TODO: Luu could probably be replaced with covar.root_decomposition()
        #   Check this!
        basis = KernelBasis(kernel=model.covar_module, centers=Z)

        def w_init(shape):
            r"""
            Initializes the weights via a Cholesky solve on the priors

            Args:
                shape: the shape of the weights

            Returns:
                weights of `shape`
            """
            prior_f = prior_fn(Z)
            prior_u = prior_f + (sigma2 ** 0.5) * torch.randn(
                prior_f.shape, dtype=prior_f.dtype, device=sigma2.device
            )
            init = torch.cholesky_solve(u - prior_u, delazify(Luu))
            init = torch.conj(init).permute(*range(init.dim() - 2), -1, -2)
            assert tuple(init.shape) == tuple(shape)
            return init

        weights = w_init(shape=list(sample_shape) + input_batch_shape + [1, m])
        return BayesianLinearSampler(
            basis=basis, weights=weights, weight_initializer=w_init
        )

    prior_fn = _create_prior_fn()
    update_fn = _create_update_fn()

    def list_add(tensors: List[Tensor]):
        return torch.add(*tensors)

    return CompositeSampler(
        join_rule=list_add,
        samplers=[prior_fn, update_fn],
        sample_shape=sample_shape,
        mean_function=model.mean_module,
        input_batch_shape=input_batch_shape,
    )
