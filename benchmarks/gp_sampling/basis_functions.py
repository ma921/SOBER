import torch
from torch import Tensor
from torch.nn import Module
import math

import gpytorch
from gpytorch.kernels import Kernel, ScaleKernel
from gpytorch.lazy import delazify

from typing import Union, Callable, Optional, Iterable

from .utils.layer import Layer


class Basis(Module):
    r"""Abstract base class for basis functions."""

    def reset_random_variables(self):
        pass


class KernelBasis(Basis):
    def __init__(self, kernel: Kernel, centers: Tensor) -> None:
        r"""
        Initialize the basis with the given centers.

        Args:
            centers: `m x d` Tensor of centers.
        """
        super().__init__()
        self.kernel = kernel
        self.centers = centers

    def forward(self, X: Tensor, **kwargs) -> Tensor:
        r"""
        Evaluates the kernel with X and centers

        Args:
            X: `n x d` tensor of points to evaluate
            **kwargs: passed to the kernel

        Returns:
            k(X, centers)
        """
        return delazify(self.kernel(X, self.centers, **kwargs))


class RandomFourierBasis(Basis):
    def __init__(
        self,
        kernel: ScaleKernel,
        units: int,
        input_batch_shape: Optional[Iterable] = None,
        activation: Callable = None,
    ) -> None:
        """
        Random Fourier Basis that is used to construct the prior component of the
        decoupled samplers.

        Note: The term 'kernel' has two different uses here. Within the context
        of the layer, it refers to the weights. Otherwise, it denotes a
        kernel function such as a member of the Matern family.

        Args:
            kernel: The kernel to approximate
            units: Number of Fourier basis
            input_batch_shape: The batch shape of the GP model. If not given, all
                batch models use the same basis, which is not ideal.
            activation: The activation function for the layer, defaults to torch.cos
        """
        super().__init__()
        self.kernel = kernel
        self.layer = None
        self.units = units
        self.input_batch_shape = (
            list() if input_batch_shape is None else list(input_batch_shape)
        )
        self.activation = torch.cos if activation is None else activation
        # TODO: implement a device / dtype option
        #   We ca just take these inputs from the model

    def forward(self, X: Tensor) -> Tensor:
        r"""
        Evaluates the Fourier basis on the input

        Args:
            X: `input_batch_shape x n x d` tensor of input values

        Returns:
            `input_batch_shape x n x units` tensor of evaluations
        """
        if self.layer is None:
            self.layer = Layer(
                units=self.units,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )
        scaled = torch.div(X, self.kernel.base_kernel.lengthscale.expand_as(X))
        outputs = self.layer(scaled)
        return (
            torch.sqrt(torch.tensor(2.0) * self.kernel.outputscale / self.units)
            * outputs
        )

    def bias_initializer(
        self,
        shape: Union[list, tuple, torch.Size],
        maxval: float = 2 * math.pi,
        **kwargs
    ) -> Tensor:
        r"""
        Provides the random samples for initializing the layer bias.

        Args:
            shape: The shape of the random samples
            maxval: The upper bound of the random samples
            **kwargs: To be passed on to `torch.rand`

        Returns:
            The random samples of `input_batch_shape x 1 x shape`
        """
        return torch.rand(size=self.input_batch_shape + [1] + shape, **kwargs) * maxval

    def kernel_initializer(
        self, shape: Union[list, tuple, torch.Size], **kwargs
    ) -> Tensor:
        r"""
        Provides the random samples for initializing the layer kernel.

        Args:
            shape: The shape of the random samples
            **kwargs: To be passed on to `torch.rand` and `randn`

        Returns:
            The random samples of `input_batch_shape x shape`
        """
        out_shape = self.input_batch_shape + list(shape)
        if isinstance(self.kernel.base_kernel, gpytorch.kernels.RBFKernel):
            return torch.randn(out_shape, **kwargs)
        elif isinstance(self.kernel.base_kernel, gpytorch.kernels.MaternKernel):
            nu = self.kernel.base_kernel.nu
            normal_rvs = torch.randn(out_shape, **kwargs)
            gamma_dist = torch.distributions.Gamma(nu, nu)
            gamma_rvs = gamma_dist.rsample(out_shape[:-2] + [1] + out_shape[-1:]).to(
                normal_rvs
            )
            return torch.rsqrt(gamma_rvs) * normal_rvs
        else:
            raise NotImplementedError

    def reset_random_variables(self) -> None:
        r"""
        Resets the random variables used in the layer by forcing re-build.
        """
        self.layer.built = False
