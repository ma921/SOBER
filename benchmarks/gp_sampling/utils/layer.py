import torch
from torch.nn import Module
from torch import Tensor
from typing import Optional, Callable


class Layer(Module):
    """
    The layer for RandomFourierBasis, minimally implemented based on tf.keras.Dense.

    Implements the operation: `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer.

    Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Callable],
        kernel_initializer: Callable,
        bias_initializer: Callable,
        **kwargs
    ) -> None:
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)

        super().__init__()
        self.units = int(units)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        # init everything here
        self.kernel = None
        self.bias = None
        self.built = False

    def build(self, input_shape):
        self.kernel = self.kernel_initializer([input_shape[-1], self.units])
        self.bias = self.bias_initializer([self.units])
        self.built = True

    def forward(self, X: Tensor):
        r"""
        Apply the layer operations to the input.

        Args:
            X: `input_batch_shape x n x d` tensor of inputs

        Returns:
            `input_batch_shape x n x units` tensor
        """
        if not self.built or self.bias.shape[-1] != self.units:
            self.build(X.shape)
        # TODO: eliminate .to calls after implementing device options
        outputs = torch.matmul(X, self.kernel.to(X))
        outputs = outputs + self.bias.expand_as(outputs).to(X)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
