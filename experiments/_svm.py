import os
import math
import torch
import pandas as pd
import numpy as np

from torch import Tensor
from torch.nn import Module
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import gpytorch.settings as gpt_settings
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize
from botorch.test_functions.base import BaseTestProblem, MultiObjectiveTestProblem
from botorch.utils.transforms import normalize, unnormalize

from sklearn.svm import SVR
from torch import Tensor
from xgboost import XGBRegressor
from SOBER._prior import MixedBinaryPrior


class DiscreteTestProblem(BaseTestProblem):
    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        integer_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__(negate=negate, noise_std=noise_std)
        self._setup(
            integer_indices=integer_indices, categorical_indices=categorical_indices
        )

    def _setup(
        self,
        integer_indices: Optional[List[int]] = None,
        categorical_indices: Optional[List[int]] = None,
    ) -> None:
        dim = self.bounds.shape[-1]
        discrete_indices = []
        if integer_indices is None:
            integer_indices = []
        if categorical_indices is None:
            categorical_indices = []
        self.register_buffer(
            "_orig_integer_indices", torch.tensor(integer_indices, dtype=torch.long)
        )
        discrete_indices.extend(integer_indices)
        self.register_buffer(
            "_orig_categorical_indices",
            torch.tensor(sorted(categorical_indices), dtype=torch.long),
        )
        discrete_indices.extend(categorical_indices)
        if len(discrete_indices) == 0:
            raise ValueError("Expected at least one discrete feature.")
        cont_indices = sorted(list(set(range(dim)) - set(discrete_indices)))
        self.register_buffer(
            "_orig_cont_indices",
            torch.tensor(
                cont_indices,
                dtype=torch.long,
                device=self.bounds.device,
            ),
        )
        self.register_buffer("_orig_bounds", self.bounds.clone())
        # remap inputs so that categorical features come after all of
        # the ordinal features
        remapper = torch.zeros(
            self.bounds.shape[-1], dtype=torch.long, device=self.bounds.device
        )
        reverse_mapper = remapper.clone()
        for i, orig_idx in enumerate(
            cont_indices + integer_indices + categorical_indices
        ):
            remapper[i] = orig_idx
            reverse_mapper[orig_idx] = i
        self.register_buffer("_remapper", remapper)
        self.register_buffer("_reverse_mapper", reverse_mapper)
        self.bounds = self.bounds[:, remapper]
        self.register_buffer("cont_indices", reverse_mapper[cont_indices])
        self.register_buffer("integer_indices", reverse_mapper[integer_indices])
        self.register_buffer("categorical_indices", reverse_mapper[categorical_indices])

        self.effective_dim = (
            self.cont_indices.shape[0]
            + self.integer_indices.shape[0]
            + int(sum(self.categorical_features.values()))
        )

        one_hot_bounds = torch.zeros(
            2, self.effective_dim, dtype=self.bounds.dtype, device=self.bounds.device
        )
        one_hot_bounds[1] = 1
        one_hot_bounds[:, self.integer_indices] = self.integer_bounds
        one_hot_bounds[:, self.cont_indices] = self.cont_bounds
        self.register_buffer("one_hot_bounds", one_hot_bounds)

    def forward(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the function on a set of points.
        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: If `True`, add observation noise as specified by `noise_std`.
        Returns:
            A `batch_shape`-dim tensor of function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        # remap to original space
        X = X[..., self._reverse_mapper]
        f = self.evaluate_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        if self.negate:
            f = -f
        return f if batch else f.squeeze(0)

    def evaluate_slack(self, X: Tensor, noise: bool = True) -> Tensor:
        r"""Evaluate the constraint function on a set of points.
        Args:
            X: A `batch_shape x d`-dim tensor of point(s) at which to evaluate the
                function.
            noise: If `True`, add observation noise as specified by `noise_std`.
        Returns:
            A `batch_shape x n_constraints`-dim tensor of function evaluations.
        """
        batch = X.ndimension() > 1
        X = X if batch else X.unsqueeze(0)
        # remap to original space
        X = X[..., self._reverse_mapper]
        f = self.evaluate_slack_true(X=X)
        if noise and self.noise_std is not None:
            f += self.noise_std * torch.randn_like(f)
        return f if batch else f.squeeze(0)

    @property
    def integer_bounds(self) -> Optional[Tensor]:
        if self.integer_indices is not None:
            return self.bounds[:, self.integer_indices]
        return None

    @property
    def cont_bounds(self) -> Optional[Tensor]:
        if self.cont_indices is not None:
            return self.bounds[:, self.cont_indices]
        return None

    @property
    def categorical_bounds(self) -> Optional[Tensor]:
        if self.categorical_indices is not None:
            return self.bounds[:, self.categorical_indices]
        return None

    @property
    def categorical_features(self) -> Optional[Dict[int, int]]:
        # Return dictionary mapping indices to cardinalities
        if self.categorical_indices is not None:
            categ_bounds = self.categorical_bounds
            return OrderedDict(
                zip(
                    self.categorical_indices.tolist(),
                    (categ_bounds[1] - categ_bounds[0] + 1).long().tolist(),
                )
            )
        return None

    @property
    def objective_weights(self) -> Optional[Tensor]:
        return None

    @property
    def is_moo(self) -> bool:
        return isinstance(self, MultiObjectiveTestProblem) and (
            self.objective_weights is None
        )

def process_uci_data(
    data: np.ndarray, n_features: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # The slice dataset can be downloaded from: https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis

    # Get the input data
    X = data[:, :-1]
    X -= X.min(axis=0)
    X = X[:, X.max(axis=0) > 1e-6]  # Throw away constant dimensions
    X = X / (X.max(axis=0) - X.min(axis=0))
    X = 2 * X - 1
    assert X.min() == -1 and X.max() == 1

    # Standardize targets
    y = data[:, -1]
    y = (y - y.mean()) / y.std()

    # Only keep 10,000 data points and n_features features
    shuffled_indices = np.random.RandomState(0).permutation(X.shape[0])[
        :10000
    ]  # Use seed 0
    X, y = X[shuffled_indices], y[shuffled_indices]

    # Use Xgboost to figure out feature importances and keep only the most important features
    xgb = XGBRegressor(max_depth=8, random_state=0).fit(X, y)
    inds = (-xgb.feature_importances_).argsort()
    X = X[:, inds[:n_features]]

    # Train/Test split on a subset of the data
    train_n = int(math.floor(0.50 * X.shape[0]))
    train_x, train_y = X[:train_n], y[:train_n]
    test_x, test_y = X[train_n:], y[train_n:]

    return train_x, train_y, test_x, test_y


class SVMFeatureSelection(DiscreteTestProblem):
    def __init__(
        self,
        dim: int,
        data: np.ndarray,
        noise_std: Optional[float] = None,
        negate: bool = False,
    ) -> None:
        n_features = dim - 3
        self.train_x, self.train_y, self.test_x, self.test_y = process_uci_data(
            data=data, n_features=n_features
        )
        self.n_features = n_features
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        super().__init__(
            negate=negate, noise_std=noise_std, integer_indices=list(range(n_features))
        )

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.tensor(
            [self._evaluate_true(x.numpy()) for x in X.view(-1, self.dim).cpu()],
            dtype=X.dtype,
            device=X.device,
        ).view(X.shape[:-1])

    def _evaluate_true(self, x: np.ndarray):
        assert x.shape == (self.dim,)
        assert (x >= self.bounds[0].cpu().numpy()).all() and (
            x <= self.bounds[1].cpu().numpy()
        ).all()
        #assert (
        #    (x[: self.n_features] == 0) | (x[: self.n_features] == 1)
        #).all()  # Features must be 0 or 1
        inds_selected = np.where(x[: self.n_features] == 1)[0]
        if inds_selected.shape[0] == 0:
            # if no features, use the mean prediction
            pred = self.train_y.mean(axis=0)
        else:
            epsilon = 0.01 * 10 ** (2 * x[-3])  # Default = 0.1
            C = 0.01 * 10 ** (4 * x[-2])  # Default = 1.0
            gamma = (
                (1 / self.n_features) * 0.1 * 10 ** (2 * x[-1])
            )  # Default = 1.0 / self.n_features
            model = SVR(C=C, epsilon=epsilon, gamma=gamma)
            model.fit(self.train_x[:, inds_selected], self.train_y)
            pred = model.predict(self.test_x[:, inds_selected])
        mse = ((pred - self.test_y) ** 2).mean(axis=0)
        return 1 * math.sqrt(mse)  # Return RMSE

def setup_svm():
    """
    Set up the experiments with SVM task
    
    Return:
    - prior: class, the function of mixed prior
    - TestFunction: class, the function that returns true Ackley function value
    """
    n_dims_cont = 3  # number of dimensions for continuous variables
    n_dims_binary = 20  # number of dimensions for binary variables
    n_dims = n_dims_cont + n_dims_binary  # total number of dimensions
    _min, _max = 0, 1  # the lower and upper bound of continous varibales
    
    SVM_DIR_NAME = './experiments/dataset/' #'../experiments/dataset/'
    data_filename = 'slice_localization_data.csv'
    # The slice dataset can be downloaded from: https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis
    data_path = os.path.join(SVM_DIR_NAME, data_filename)
    df = pd.read_csv(data_path)
    data = np.array(df)
    svm = SVMFeatureSelection(n_dims, data)
    
    def TestFunction(X):
        eval_ = svm(X)
        return -1 * eval_.squeeze()

    prior = MixedBinaryPrior(n_dims_cont, n_dims_binary, _min, _max, continous_first=False)
    
    return prior, TestFunction
