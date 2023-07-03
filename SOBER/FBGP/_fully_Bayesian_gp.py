import copy
import torch
import gpytorch
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from ._scale_vbq import ScaleVanillaGP
from .._gp import ExactGPModel
from .._utils import Utils
from .._kernel import Kernel
from .._rchq import recombination
from .._weights import WeightsStabiliser

class ManagingGPHyperparameters:
    def __init__(self, rng=100):
        """
        Managing Gaussian process hyperparameter
        
        Args:
        - rng: int, the range of the likelihood bounds
        """
        self.rng = rng
        
    def show_hypers(self, model):
        """
        Show the current hyperparameter values
        
        Args:
        - model: gpytorch.models, the Gaussian process model
        """
        print(
            "lik_var: " + str(model.likelihood.noise.item()) + '\n' +
            "lengthscale: " + str(model.covar_module.base_kernel.lengthscale.item())+ '\n' +
            "outputscale: " + str(model.covar_module.outputscale.item())
        )
        
    def extract_hypers(self, model):
        """
        Return the current hyperparameters.
        
        Args:
        - model: gpytorch.models, the Gaussian process model
        
        Return:
        - Theta: torch.tensor, the current hyperparameters
        """
        return torch.tensor([
            model.likelihood.noise.detach(),
            model.covar_module.base_kernel.lengthscale.detach(),
            model.covar_module.outputscale.detach(),
        ])

    def set_hypers(self, model, theta):
        """
        Update the GP hyperparameters.
        
        Args:
        - model: gpytorch.models, the Gaussian process model
        - theta: torch.tensor, the hyperparameters to set
        
        Return:
        - model: gpytorch.models, the updated Gaussian process model
        """
        hypers = {
            'likelihood.noise_covar.noise': theta[0],
            'covar_module.base_kernel.lengthscale': theta[1],
            'covar_module.outputscale': theta[2],
        }
        model.initialize(**hypers)
        return model

    def reset_GP(self, train_x, train_y, theta):
        """
        Reset the GP model.
        
        Args:
        - train_x: torch.tensor, the input
        - train_y: torch.tensor, the output
        - theta: torch.tensor, the hyperparameters to set
        
        Return:
        - model: gpytorch.models, the updated Gaussian process model
        """
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise_covar.register_constraint(
            "raw_noise",
            gpytorch.constraints.Interval(theta[0] / self.rng, theta[0] * self.rng)
        )
        gp_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        model = ExactGPModel(train_x, train_y, likelihood, gp_kernel)
        model = self.set_hypers(model, theta)
        return model
    
class LogMarginalLikelihood(ManagingGPHyperparameters):
    def __init__(self, gp):
        """
        Log marginal likelihood of FITBO model
        
        Args:
        - gp: FitboGP class, the function of underlying model
        """
        super().__init__(rng=100) # inherit ManagingGPHyperparameters class
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.utils = Utils(device)
        self.eps = -torch.sqrt(torch.tensor(torch.finfo().max))
        self.Xobs = copy.deepcopy(gp.model.train_inputs[0])
        self.ymax = copy.deepcopy(gp.model.train_targets.max())
        self.eta = copy.deepcopy(gp.alpha)
        self.fobs = gp.Y_unwarp #(gp.Y_unwarp - gp.Y_unwarp.mean()) / gp.Y_unwarp.std()
        self.n_data = self.Xobs.size(0)
        self.theta_map = self.extract_hypers(gp.model)
        
    def log_to_exp_transform(self, theta):
        """
        Transform back the hyperparameters to the original space
        
        Args:
        - theta: torch.tensor, the log-transformed hyperparameters
        """
        Theta = theta.exp()
        if Theta.dim() == 1:
            Theta[0] = self.eta + Theta[0]
        else:
            Theta[:,0] = self.eta + Theta[:,0]
        return Theta

    def mll(self, Theta):
        """
        Compute the marginal log likelihood of FITBO model
        
        Args:
        - Theta: torch.tensor, the hyperparameters
        
        Return:
        - mll: torch.tensor, the marginal log likelihood
        """
        # hypers
        eta = Theta[0]
        theta = Theta[1:]

        # reset hypers
        gobs_pseudo = eta.sign() * (2*(eta - self.fobs)).sqrt()
        model = self.reset_GP(self.Xobs, gobs_pseudo, theta)
        model.eval()

        # g space prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            function_dist = model(self.Xobs)
            output = model.likelihood(function_dist)
        mu_g = output.loc
        var_g = output.variance
        covar_g = output.covariance_matrix

        # f space prediction
        mu_f = eta - 0.5 * (mu_g**2 + var_g)
        covar_f = mu_g.unsqueeze(1) * covar_g * mu_g.unsqueeze(0) + 0.5 * (covar_g ** 2)
        res = MultivariateNormal(
            mu_f.detach(), 
            self.utils.make_cov_psd(covar_f),
        ).log_prob(self.fobs)
        mll = res.div_(self.n_data)
        return mll
    
    def __call__(self, theta):
        """
        Compute the marginal log likelihood at given log-transformed hyperparamters
        
        Args:
        - theta: torch.tensor, the log-transformed hyperparameters
        
        Return:
        - mll: torch.tensor, the marginal log likelihood
        """
        Theta = self.log_to_exp_transform(theta)
        try:
            return self.mll(Theta)
        except:
            return self.eps

def sampling_hypers(model, hyperprior, n_hypers=1000, use_map=False):
    """
    Sampling hypersamples with fully Bayesian Gaussian process (FBGP) model

    Args:
    - model: FitboGP class, the function of underlying model
    - hyperprior: RBFHyperPrior class, the function of hyperprior
    - n_hypers: int, the number of hypersamples from hyperprior
    - use_map: bool, use MAP estimated hypersamples as the mean if true, otherwise not.

    Return:
    - Hypersamples: torch.tensor, the hypersamples from hyperprior
    - LMLs: torch.tensor, the computed log marginal likelihoods
    """
    lml = LogMarginalLikelihood(model)
    if use_map:
        hyperprior.initialise(lml.theta_map)
    hypersamples = hyperprior.sample(n_hypers)
    hypersamples = torch.vstack([
        torch.cat([torch.tensor([-10]), lml.theta_map.log()]),
        hypersamples,
    ])
    LMLs = torch.tensor([lml(theta) for theta in hypersamples])
    Hypersamples = lml.log_to_exp_transform(hypersamples)
    return Hypersamples, LMLs

def quadrature_distillation(Hypersamples, LMLs, gp_kernel, n_nys=100, n_qd=50):
    """
    Quadrature distillation for sparsifying weighted hypersamples

    Args:
    - Hypersamples: torch.tensor, the hypersamples from hyperprior
    - LMLs: torch.tensor, the computed log marginal likelihoods
    - gp_kernel: gpytorch.kernels, the function of kernel for Bayesian quadrature model
    - n_nys: int, the number of hypersamples for Nyström approximation
    - n_qd: int, the number of resulting distilled hypersamples

    Return:
    - w_qd: torch.tensor, the non-negative weights for distilled hypersamples
    - Theta_qd: torch.tensor, the distilled hypersamples
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # hyperposterior as weighted hyperprior samples
    n_rec = len(Hypersamples)
    weights = (LMLs - LMLs.max()).exp()
    
    W = WeightsStabiliser()
    weights = W.cleansing_weights(weights)
    idx_nys = W.deweighted_resampling(weights, n_nys)
    Hyper_nys = Hypersamples[idx_nys]
    
    # modelling hyper-GP
    VBQ = ScaleVanillaGP(Hypersamples, LMLs, gp_kernel, device)
    kernel = Kernel(VBQ.model, mode="kernel")
    
    # recombination
    idx, w_qd = recombination(
        Hypersamples,
        Hyper_nys,
        n_qd,
        kernel,
        device,
        init_weights=weights,
    )
    Theta_qd = Hypersamples[idx]
    return w_qd, Theta_qd

class FullyBayesianGP(LogMarginalLikelihood):
    def __init__(self, gp, w_qd, Theta_qd):
        """
        Fully Bayesian Gaussian process model

        Args:
        - gp: FitboGP class, the function of underlying model
        - w_qd: torch.tensor, the non-negative weights for distilled hypersamples
        - Theta_qd: torch.tensor, the distilled hypersamples
        """
        super().__init__(gp) # inherit LogMarginalLikelihood class
        self.w_qd = w_qd.detach()
        self.Theta_qd = Theta_qd.detach()
        self.is_fbgp = True
    
    def fitbo_predict(self, x_test, Theta):
        """
        Posterior predictive distribution from FITBO GP model

        Args:
        - x_test: torch.tensor, the input
        - Theta: torch.tensor, the set of hypersamples
        
        Return:
        - mu_f: torch.tensor, the posterior predictive mean
        - var_f: torch.tensor, the posterior predictive variance
        """
        gobs_pseudo = Theta[0].sign() * (2*(Theta[0] - self.fobs)).sqrt()
        model = self.reset_GP(self.Xobs, gobs_pseudo, Theta[1:])
        
        # g space prediction
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            function_dist = model(x_test)
            output = model.likelihood(function_dist)
        mu_g = output.loc
        var_g = output.variance
        var_g[var_g < 0] = 0

        # f space prediction
        mu_f = Theta[0] - 0.5 * (mu_g**2 + var_g)
        var_f = mu_g * var_g * mu_g + 0.5 * (var_g ** 2)
        return mu_f, var_f

    def fitbo_predict_batch(self, x_test, Theta):
        """
        A batch of posterior predictive distribution from FITBO GP model

        Args:
        - x_test: torch.tensor, the input
        - Theta: torch.tensor, the set of hypersamples
        
        Return:
        - mu_batch: torch.tensor, the batch of posterior predictive mean
        - var_batch: torch.tensor, the batch of  posterior predictive variance
        """
        mu, var = self.fitbo_predict(x_test, Theta)
        var[var < 0] = 0
        return torch.stack([mu, var])

    def batch_predict(self, x_test):
        """
        A batch of posterior predictive distribution from FITBO GP model

        Args:
        - x_test: torch.tensor, the input
        
        Return:
        - mu_batch: torch.tensor, the batch of posterior predictive mean
        - var_batch: torch.tensor, the batch of  posterior predictive variance
        """
        batch_pred = torch.stack([
        self.fitbo_predict_batch(x_test, Theta)
            for Theta in self.Theta_qd
        ])
        mu_batch, var_batch = batch_pred[:,0,:], batch_pred[:,1,:]
        return mu_batch, var_batch
    
    def marginal_predict(self, x_test):
        """
        Marginal posterior predictive distribution from FITBO GP model

        Args:
        - x_test: torch.tensor, the input
        
        Return:
        - mu_fbgp: torch.tensor, the marginal posterior predictive mean
        - var_fbgp: torch.tensor, the marginal posterior predictive variance
        """
        mu_batch, var_batch = self.batch_predict(x_test)
        mu_fbgp = self.w_qd @ mu_batch
        var_fbgp = self.w_qd @ (var_batch + mu_batch.pow(2)) - (self.w_qd @ mu_batch).pow(2)
        return mu_fbgp, var_fbgp
    
    def marginal_predictive_mean(self, x_test):
        """
        Marginal posterior predictive mean from FITBO GP model

        Args:
        - x_test: torch.tensor, the input
        
        Return:
        - mu_fbgp: torch.tensor, the marginal posterior predictive mean
        """
        mu_batch, _ = self.batch_predict(x_test)
        return self.w_qd @ mu_batch
    
    def marginal_predictive_covariance(self, x_test, y_test):
        """
        Marginal posterior predictive covariance from FITBO GP model

        Args:
        - x_test: torch.tensor, the input
        - y_test: torch.tensor, the input
        
        Return:
        - cov_ny: torch.tensor, the marginal posterior predictive covariance
        """
        mu_x, _ = self.batch_predict(x_test)
        mu_y, _ = self.batch_predict(y_test)
        Ex = self.w_qd @ mu_x
        Ey = self.w_qd @ mu_y
        W = 1/(1 - self.w_qd.pow(2).sum())
        cov_xy = W * (self.w_qd.unsqueeze(1) * (mu_x - Ex.unsqueeze(0))).T @ (mu_y - Ey.unsqueeze(0))
        return cov_xy
