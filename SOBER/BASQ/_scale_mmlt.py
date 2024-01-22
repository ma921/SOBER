import copy
import torch
from .._gp import update_gp, predict, predictive_covariance
from .._utils import Utils


class ScaleMmltGP(Utils):
    def __init__(
        self,
        Xobs,
        Yobs,
        gp_kernel,
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
        Scale MMLT BQ modelling
        Scale MMLT is a doubly warped GP modelling. It consists of three levels of warped space.
        The observation y is assumed to belong to g space. (query returns true log likelihoods.)
        WSABI modelling permits a factorisation trick in log-space.
         _________________________ _______________________________________ __________________
        |          f space        |                g space                |     h space      |
        |_________________________|_______________________________________|__________________|
        |             f           |               f / exp(β)              |  h = log(g + 1)  |
        |        f = g exp(β)     |             g = exp(h) - 1            |        h         |
        |_________________________|_______________________________________|__________________|
        |      f = GP(μ_f, σ_f)   |           g = GP(μ_g, σ_g)            | h = GP(μ_h, σ_h) |
        |     μ_f = μ_g(x)exp(β)  |        μ_g = exp(μ_h + 1/2 σ_h) - 1   |        μ_h       |
        |   σ_f = σ_g(x,y)exp(2β) | σ_g = μ_g(x)μ_g(y)(exp(σ_h(x,y)) - 1) |        σ_h       |
        |_________________________|_______________________________________|__________________|

        ScaleMmltGP class summarises the functions of training, updating the warped GP model.
        This also provides the prediction and kernel of WSABI GP.
        The modelling of WSABI-L and WSABI-M can be easily switched by changing "label".
        The above table is the case of WSABI-M.

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
        super().__init__()
        self.gp_kernel = gp_kernel
        self.alpha_factor = 1
        self.alpha = alpha_factor
        self.lik = lik
        self.training_iter = training_iter
        self.thresh = thresh
        self.lr = lr
        self.rng = rng
        self.train_lik = train_lik
        self.optimiser = optimiser
        self.is_bq = True

        self.jitter = self.tensor(0)  # 1e-6
        self.Y_log = copy.deepcopy(Yobs)

        self.model = update_gp(
            Xobs,
            self.process_y_warping_with_scaling(Yobs),
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

    def process_y_warping_with_scaling(self, y_obs):
        """
        Args:
           - y_obs: torch.tensor, observations of true_loglikelihood

        Returns:
           - y_h: torch.tensor, warped observations in h space that contains no anomalies and the updated alpha hyperparameter.
        """

        y = self.remove_anomalies(y_obs)
        self.beta = torch.max(y)
        y_g = torch.exp(y - self.beta)
        y_h = self.warp_from_g_to_h(y_g)
        return y_h

    def warp_from_g_to_h(self, y_g):
        """
        Args:
           - y_f: torch.tensor, warped observations in g space

        Returns:
           - y_g: torch.tensor, warped observations in h space
        """
        y_h = torch.log(y_g + 1)
        return y_h

    def unwarp_from_h_to_g(self, y_h):
        """
        Args:
           - y_f: torch.tensor, warped observations in h space

        Returns:
           - y_g: torch.tensor, warped observations in g space
        """
        y_g = torch.exp(y_h) - 1
        return y_g
    

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
        Yall_h = self.process_y_warping_with_scaling(Yall_log)
        return Xall, Yall_h

    def update_mmlt_gp_with_scaling(self, X, Y):
        """
        Args:
           - X: torch.tensor, X samples to be added to the existing data Xobs
           - Y: torch.tensor, unwarped Y observations to be added to the existing data Yobs
        """
        X_h, Y_h = self.cat_observations_with_scaling(X, Y)
        self.model = update_gp(
            X_h,
            Y_h,
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

    def retrain_gp_with_scaling(self):
        X_h = self.model.train_inputs[0]
        Y_h = self.process_y_warping_with_scaling(copy.deepcopy(self.Y_log))
        self.model = update_gp(
            X_h,
            Y_h,
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

    def hspace_predict(self, x):
        """
        Args:
           - x: torch.tensor, x locations to be predicted

        Returns:
           - mu_h: torch.tensor, unwarped predictive mean in h space at given locations x.
           - var_h: torch.tensor, unwarped predictive variance in h space at given locations x.
        """
        mu_h, var_h = predict(x, self.model)
        return mu_h, var_h

    def gspace_predict(self, x):
        """
        Args:
           - x: torch.tensor, x locations to be predicted

        Returns:
           - mu_g: torch.tensor, unwarped predictive mean in g space at given locations x.
           - var_g: torch.tensor, unwarped predictive variance in g space at given locations x.
        """
        mu_h, var_h = self.hspace_predict(x)
        mu_g = (mu_h + 0.5 * var_h).exp() - 1
        var_g = (mu_g ** 2) * (var_h.exp() - 1)
        return mu_g, var_g

    def hspace_mean_predict(self, x):
        """
        Args:
           - x: torch.tensor, x locations to be predicted

        Returns:
           - mu_h: torch.tensor, unwarped predictive mean in h space at given locations x.
        """
        mu_h, _ = self.hspace_predict(x)
        return mu_h

    def gspace_mean_predict(self, x):
        """
        Args:
           - x: torch.tensor, x locations to be predicted

        Returns:
           - mu_g: torch.tensor, unwarped predictive mean in g space at given locations x.
        """
        mu_g, _ = self.gspace_predict(x)
        return mu_g

    def hspace_kernel(self, x, y):
        """
        Args:
           - x: torch.tensor, x locations to be predicted
           - y: torch.tensor, y locations to be predicted

        Args:
           - CLy: torch.tensor, the positive semi-definite Gram matrix of predictive variance in hscape
        """
        return predictive_covariance(x, y, self.model)

    def gspace_kernel(self, x, y):
        """
        Args:
           - x: torch.tensor, x locations to be predicted
           - y: torch.tensor, y locations to be predicted

        Returns:
           - CLy: torch.tensor, the positive semi-definite Gram matrix of predictive variance in gscape
        """
        mu_g_x = self.gspace_mean_predict(x)
        mu_g_y = self.gspace_mean_predict(y)
        cov_h_xy = self.hspace_kernel(x, y)
        if len(cov_h_xy.shape) == 2:
            CLy = mu_g_x.unsqueeze(1) * mu_g_y.unsqueeze(0) * (cov_h_xy.exp() - 1)
        elif len(cov_h_xy.shape) == 3:
            CLy = mu_g_x.unsqueeze(1).unsqueeze(0) * mu_g_y.unsqueeze(1) * (cov_h_xy.exp() - 1)

        d = min(len(x), len(y))
        CLy[range(d), range(d)] = CLy[range(d), range(d)] + self.jitter
        return CLy
