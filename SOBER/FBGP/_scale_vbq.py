import copy
import torch
from .._gp import update_gp, predict, predictive_covariance
from .._utils import Utils


class ScaleVanillaGP:
    def __init__(
        self,
        Xobs,
        Yobs,
        gp_kernel,
        device,
        lik=1e-10,
        training_iter=10000,
        thresh=0.01,
        lr=0.1,
        rng=10,
        train_lik=False,
        optimiser="BoTorch",
    ):
        """
        Vanilla Gaussian process modelling for Bayesian quadrature.
        The observed Y (Yobs) is assumed to be non-normalised log-likelihood.
        "Scale" here means this model automatically normalise the log-likelihood for vanilla GP.
        
        Args:
           - Xobs: torch.tensor, X samples, X belongs to prior measure.
           - Yobs: torch.tensor, Y observations, Y = true_likelihood(X).
           - gp_kernel: gpytorch.kernels, GP kernel function
           - device: torch.device, device, cpu or cuda
           - lik: float, the initial value of GP likelihood noise variance
           - train_iter: int, the maximum iteration for GP hyperparameter training.
           - thresh: float, the threshold as a stopping criterion of GP hyperparameter training.
           - lr: float, the learning rate of Adam optimiser
           - rng: int, tne range coefficient of GP likelihood noise variance
           - train_like: bool, flag whether or not to update GP likelihood noise variance
        """
        self.gp_kernel = gp_kernel
        self.device = device
        self.lik = lik
        self.training_iter = training_iter
        self.thresh = thresh
        self.lr = lr
        self.rng = rng
        self.train_lik = train_lik
        self.optimiser = optimiser

        self.jitter = 1e-6
        self.Y_log = copy.deepcopy(Yobs)
        self.utils = Utils(device)

        self.model = update_gp(
            Xobs,
            self.process_y_with_scaling(Yobs),
            gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter,
            thresh=self.thresh,
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
            optimiser=self.optimiser,
        )
    
    def process_y_with_scaling(self, y_obs):
        """
        Args:
           - y_obs: torch.tensor, observations of true_loglikelihood

        Returns:
           - y_h: torch.tensor, warped observations in h space that contains no anomalies and the updated alpha hyperparameter.
        """

        y = self.utils.remove_anomalies(y_obs)
        self.beta = torch.max(y)
        y_exp = torch.exp(y - self.beta)
        return y_exp

    def cat_observations_with_scaling(self, X, Y):
        """
        Args:
           - X: torch.tensor, X samples to be added to the existing data Xobs
           - Y: torch.tensor, unwarped Y observations to be added to the existing data Yobs

        Returns:
           - Xall: torch.tensor, X samples that contains all samples
           - Yall: torch.tensor, warped Y observations that contains all observations
        """
        Xobs = self.model.train_inputs[0]
        Yobs_log = copy.deepcopy(self.Y_log)
        if len(self.model.train_targets.shape) == 0:
            Yobs_log = Yobs_log.unsqueeze(0)
        Xall = torch.cat([Xobs, X])
        Yall_log = torch.cat([Yobs_log, Y])
        self.Y_log = copy.deepcopy(Yall_log)
        Yall_exp = self.process_y_with_scaling(Yall_log)
        return Xall, Yall_exp

    def update_gp(self, X, Y):
        """
        Args:
           - X: torch.tensor, X samples to be added to the existing data Xobs
           - Y: torch.tensor, Y observations to be added to the existing data Yobs
        """
        Xall, Yall = self.cat_observations_with_scaling(X, Y)
        self.model = update_gp(
            Xall,
            Yall,
            self.gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter,
            thresh=self.thresh,
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
            optimiser=self.optimiser,
        )

    def retrain_gp(self):
        Xobs = self.model.train_inputs[0]
        Yobs = self.model.train_targets
        self.model = update_gp(
            Xobs,
            Yobs,
            self.gp_kernel,
            self.device,
            lik=self.lik,
            training_iter=self.training_iter,
            thresh=self.thresh,
            lr=self.lr,
            rng=self.rng,
            train_lik=self.train_lik,
            optimiser=self.optimiser,
        )

    def predictive_kernel(self, x, y):
        """
        Args:
           - x: torch.tensor, x locations to be predicted
           - y: torch.tensor, y locations to be predicted

        Args:
           - CLy: torch.tensor, the positive semi-definite Gram matrix of predictive variance
        """
        return predictive_covariance(x, y, self.model)

    def predict(self, x):
        """
        Args:
           - x: torch.tensor, x locations to be predicted

        Returns:
           - mu: torch.tensor, predictive mean at given locations x.
           - var: torch.tensor, predictive variance at given locations x.
        """
        mu, var = predict(x, self.model)
        return mu, var

    def predict_mean(self, x):
        """
        Args:
           - x: torch.tensor, x locations to be predicted

        Returns:
           - mu: torch.tensor, predictive mean at given locations x.
        """
        mu, _ = predict(x, self.model)
        return mu
