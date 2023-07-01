import copy
import torch
from .._gp import set_gp, update_gp, predict, predictive_covariance
from .._utils import Utils


class FitboGP:
    def __init__(
        self,
        Xobs,
        Yobs,
        gp_kernel,
        device,
        label="wsabim",
        alpha_factor=1,
        lik=1e-10,
        training_iter=10000,
        thresh=0.01,
        lr=0.1,
        rng=10,
        train_lik=False,
        optimiser="BoTorch",
    ):
        """
        FITBO (WSABI BQ) modelling
        FITBO -> https://arxiv.org/abs/1711.00673
        WSABI -> https://arxiv.org/abs/1411.0439
        WsabiGP class summarises the functions of training, updating the warped GP model.
        This also provides the prediction and kernel of WSABI GP.
        The modelling of WSABI-L and WSABI-M can be easily switched by changing "label".

        Args:
           - Xobs: torch.tensor, X samples, X belongs to prior measure.
           - Yobs: torch.tensor, Y observations, Y = true_likelihood(X).
           - gp_kernel: gpytorch.kernels, GP kernel function
           - device: torch.device, device, cpu or cuda
           - label: string, the wsabi type, ["wsabil", "wsabim"]
           - lik: float, the initial value of GP likelihood noise variance
           - train_iter: int, the maximum iteration for GP hyperparameter training.
           - thresh: float, the threshold as a stopping criterion of GP hyperparameter training.
           - lr: float, the learning rate of Adam optimiser
           - rng: int, tne range coefficient of GP likelihood noise variance
           - train_like: bool, flag whether or not to update GP likelihood noise variance
           - optimiser: string, select the optimiser ["L-BFGS-B", "Adam"]
        """
        self.gp_kernel = gp_kernel
        self.device = device
        self.alpha_factor = alpha_factor
        self.lik = lik
        self.training_iter = training_iter
        self.thresh = thresh
        self.lr = lr
        self.rng = rng
        self.train_lik = train_lik
        self.optimiser = optimiser

        self.jitter = 0  # 1e-6
        self.Y_unwarp = copy.deepcopy(Yobs)
        self.utils = Utils(device)

        self.model = update_gp(
            Xobs,
            self.process_y_warping(Yobs),
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
        self.setting(label)

    def setting(self, label):
        """
        Args:
           - label: string, the wsabi type, ["wsabil", "wsabim"]
        """
        if label == "wsabil":
            self.kernel = self.wsabil_kernel
            self.predict = self.wsabil_predict
            self.predict_mean = self.wsabil_mean_predict
        elif label == "wsabim":
            self.kernel = self.wsabim_kernel
            self.predict = self.wsabim_predict
            self.predict_mean = self.wsabim_mean_predict

    def warp_y(self, y):
        """
        Args:
           - y: torch.tensor, observations

        Returns:
           - y: torch.tensor, warped observations
        """
        return self.alpha.sign() * torch.sqrt(2 * (self.alpha - y))

    def unwarp_y(self, y):
        """
        Args:
           - y: torch.tensor, warped observations

        Returns:
           - y: torch.tensor, unwarped observations
        """
        return self.alpha - 0.5 * (y ** 2)

    def process_y_warping(self, y):
        """
        Args:
           - y: torch.tensor, observations

        Returns:
           - y: torch.tensor, warped observations that contains no anomalies and the updated alpha hyperparameter.
        """
        y = self.utils.remove_anomalies(y)
        #y -= y.min() 
        #y = (y - y.mean()) / y.std()
        self.alpha = self.alpha_factor * torch.max(y)
        y = self.warp_y(y)
        return y

    def cat_observations(self, X, Y):
        """
        Args:
           - X: torch.tensor, X samples to be added to the existing data Xobs
           - Y: torch.tensor, unwarped Y observations to be added to the existing data Yobs

        Returns:
           - Xall: torch.tensor, X samples that contains all samples
           - Yall: torch.tensor, warped Y observations that contains all observations
        """
        Xobs = self.model.train_inputs[0]
        Yobs = copy.deepcopy(self.Y_unwarp)
        if len(self.model.train_targets.shape) == 0:
            Yobs = Yobs.unsqueeze(0)
        Xall = torch.cat([Xobs, X])
        _Yall = torch.cat([Yobs, Y])
        self.Y_unwarp = copy.deepcopy(_Yall)
        Yall = self.process_y_warping(_Yall)
        return Xall, Yall

    def update_wsabi_gp(self, X, Y):
        """
        Args:
           - X: torch.tensor, X samples to be added to the existing data Xobs
           - Y: torch.tensor, unwarped Y observations to be added to the existing data Yobs
        """
        X_warp, Y_warp = self.cat_observations(X, Y)
        self.model = update_gp(
            X_warp,
            Y_warp,
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
        """
        self.model = set_gp(
            X_warp,
            Y_warp,
            self.gp_kernel,
            self.device,
            lik=self.lik,
            rng=self.rng,
            train_lik=self.train_lik,
        )
        """

    def retrain_gp(self):
        X_warp = self.model.train_inputs[0]
        Y_warp = self.process_y_warping(copy.deepcopy(self.Y_unwarp))
        self.model = update_gp(
            X_warp,
            Y_warp,
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

    def memorise_parameters(self):
        self.likelihood_memory = copy.deepcopy(torch.tensor(self.model.likelihood.noise.item()))
        self.outputsacle_memory = copy.deepcopy(torch.tensor(self.model.covar_module.outputscale.item()))
        self.lengthscale_memory = copy.deepcopy(torch.tensor(self.model.covar_module.base_kernel.lengthscale.item()))

    def remind_parameters(self):
        hypers = {
            'likelihood.noise_covar.noise': self.likelihood_memory,
            'covar_module.outputscale': self.outputsacle_memory,
            'covar_module.base_kernel.lengthscale': self.lengthscale_memory,
        }
        self.model.initialize(**hypers)

    def predictive_kernel(self, x, y):
        """
        Args:
           - x: torch.tensor, x locations to be predicted
           - y: torch.tensor, y locations to be predicted

        Args:
           - CLy: torch.tensor, the positive semi-definite Gram matrix of predictive variance
        """
        return predictive_covariance(x, y, self.model)

    def wsabil_kernel(self, x, y):
        """
        Args:
           - x: torch.tensor, x locations to be predicted
           - y: torch.tensor, y locations to be predicted

        Returns:
           - CLy: torch.tensor, the positive semi-definite Gram matrix of WSABI-L variance
        """
        mu_x, _ = predict(x, self.model)
        mu_y, _ = predict(y, self.model)
        cov_xy = predictive_covariance(x, y, self.model)
        CLy = mu_x.unsqueeze(1) * cov_xy * mu_y.unsqueeze(0)

        d = min(len(x), len(y))
        CLy[range(d), range(d)] = CLy[range(d), range(d)] + self.jitter
        return CLy

    def wsabim_kernel(self, x, y):
        """
        Args:
           - x: torch.tensor, x locations to be predicted
           - y: torch.tensor, y locations to be predicted

        Returns:
           - CLy: torch.tensor, the positive semi-definite Gram matrix of WSABI-M variance
        """
        mu_x, _ = predict(x, self.model)
        mu_y, _ = predict(y, self.model)
        cov_xy = predictive_covariance(x, y, self.model)
        CLy = mu_x.unsqueeze(1) * cov_xy * mu_y.unsqueeze(0) + 0.5 * (cov_xy ** 2)

        d = min(len(x), len(y))
        CLy[range(d), range(d)] = CLy[range(d), range(d)] + self.jitter
        return CLy

    def wsabil_predict(self, x):
        """
        Args:
           - x: torch.tensor, x locations to be predicted

        Returns:
           - mu: torch.tensor, unwarped predictive mean at given locations x.
           - var: torch.tensor, unwarped predictive variance at given locations x.
        """
        mu_warp, var_warp = predict(x, self.model)
        mu = self.alpha - 0.5 * mu_warp**2
        var = mu_warp * var_warp * mu_warp
        return mu, var

    def wsabim_predict(self, x):
        """
        Args:
           - x: torch.tensor, x locations to be predicted

        Returns:
           - mu: torch.tensor, unwarped predictive mean at given locations x.
           - var: torch.tensor, unwarped predictive variance at given locations x.
        """
        mu_warp, var_warp = predict(x, self.model)
        mu = self.alpha - 0.5 * (mu_warp**2 + var_warp)
        var = mu_warp * var_warp * mu_warp + 0.5 * (var_warp ** 2)
        return mu, var

    def wsabil_mean_predict(self, x):
        """
        Args:
           - x: torch.tensor, x locations to be predicted

        Returns:
           - mu: torch.tensor, unwarped predictive mean at given locations x.
        """
        mu_warp, _ = predict(x, self.model)
        mu = self.alpha - 0.5 * mu_warp**2
        return mu

    def wsabim_mean_predict(self, x):
        """
        Args:
           - x: torch.tensor, x locations to be predicted

        Returns:
           - mu: torch.tensor, unwarped predictive mean at given locations x.
        """
        mu_warp, var_warp = predict(x, self.model)
        mu = self.alpha - 0.5 * (mu_warp**2 + var_warp)
        return mu
