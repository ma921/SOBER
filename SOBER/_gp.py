import torch
import warnings
import gpytorch
from botorch.fit import fit_gpytorch_mll
from gpytorch.priors.torch_priors import GammaPrior


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, gp_kernel):
        """
        Args:
            - train_x: torch.tensor, inputs. torch.Size(n_data, n_dims)
            - train_y: torch.tensor, observations
            - likelihood: gpytorch.likelihoods, GP likelihood model
            - gp_kernel: gpytorch.kernels, GP kernel model
        """
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gp_kernel

    def forward(self, x):
        """
        Args:
            - x: torch.tensor, inputs. torch.Size(n_data, n_dims)

        Returns:
            - torch.distributions, predictive posterior distribution at given x
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def set_gp(train_x, train_y, gp_kernel, device, lik=1e-10, rng=10, train_lik=False):
    """
    We can select whether or not to train likelihood variance.
    The true_likelihood query must be noiseless, so learning GP likelihood noise variance could be redundant.
    However, likelihood noise variance plays an important role in a limited number of samples in the early stage.
    So, setting interval constraints keeps the likelihood noise variance within a safe area.
    Otherwise, GP could confuse the meaningful multimodal peaks of true_likelihood as noise.

    Args:
        - train_x: torch.tensor, inputs. torch.Size(n_data, n_dims)
        - train_y: torch.tensor, observations
        - gp_kernel: gpytorch.kernels, GP kernel model
        - device: torch.device, cpu or cuda
        - lik: float, the initial value of GP likelihood noise variance
        - rng: int, tne range coefficient of GP likelihood noise variance
        - train_like: bool, flag whether or not to update GP likelihood noise variance

    Returns:
        - model: gpytorch.models, function of GP model.
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(lik / rng, lik * rng))
    model = ExactGPModel(train_x, train_y, likelihood, gp_kernel)
    model.covar_module.base_kernel.lengthscale_prior = GammaPrior(3.0, 6.0)
    model.covar_module.outputscale_prior = GammaPrior(2.0, 0.15)
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(lik),
    }

    model.initialize(**hypers)
    if not train_lik:
        model.likelihood.raw_noise.requires_grad = False

    if device.type == 'cuda':
        model = model.cuda()
        model.likelihood = model.likelihood.cuda()
    return model


class Closure:
    """
    Args:
        - mll: gpytorch.mlls.ExactMarginalLogLikelihood, marginal log likelihood
        - optimiser: torch.optim, L-BFGS-B optimizer from FullBatchLBFGS

    Returns:
        - loss: torch.tensor, negative log marginal likelihood of GP
    """
    def __init__(self, mll, optimizer):
        self.mll = mll
        self.optimizer = optimizer
        self.train_inputs, self.train_targets = mll.model.train_inputs, mll.model.train_targets

    def __call__(self):
        self.optimizer.zero_grad()
        with gpytorch.settings.fast_computations(log_prob=True):
            output = self.mll.model(*self.train_inputs)
            args = [output, self.train_targets]
            loss = -self.mll(*args).sum()
        return loss


def train_GP_with_BFGS(mll, training_iter, thresh):
    """
    L-BFGS-B implementation is from https://github.com/hjmshi/PyTorch-LBFGS

    Args:
        - mll: gpytorch.mlls.ExactMarginalLogLikelihood, marginal log likelihood
        - training_iter: int, the maximum number of iteration of optimisation loop
        - thresh: float, the stopping criterion

    Returns:
        - mll: gpytorch.mlls.ExactMarginalLogLikelihood, marginal log likelihood
    """
    # Use full-batch L-BFGS optimizer
    optimizer = FullBatchLBFGS(mll.model.parameters())
    closure = Closure(mll, optimizer)
    loss = closure()
    loss.backward()
    loss_best = torch.tensor(1e10)

    for i in range(training_iter):
        # perform step and update curvature
        options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)

        if loss.item() < loss_best:
            delta = torch.abs(loss_best - loss.detach())
            loss_best = loss.item()
            if delta < thresh:
                break
    return mll


def train_GP_with_Adam(mll, lr, training_iter, thresh):
    """
    Args:
        - mll: gpytorch.mlls.ExactMarginalLogLikelihood, marginal log likelihood
        - lr: float, the learning rate
        - training_iter: int, the maximum number of iteration of optimisation loop
        - thresh: float, the stopping criterion

    Returns:
        - mll: gpytorch.mlls.ExactMarginalLogLikelihood, marginal log likelihood
    """
    optimizer = torch.optim.Adam(mll.model.parameters(), lr=lr)
    train_x = mll.model.train_inputs[0]
    train_y = mll.model.train_targets
    loss_best = torch.tensor(1e10)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = mll.model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if loss.item() < loss_best:
            delta = torch.abs(loss_best - loss.detach())
            loss_best = loss.item()
            if delta < thresh:
                break
    return mll


def train_GP(model, training_iter=50, thresh=0.01, lr=0.1, optimiser="BoTorch"):
    """
    Args:
        - model: gpytorch.models, function of GP model.
        - train_iter: int, the maximum iteration for GP hyperparameter training.
        - thresh: float, the threshold as a stopping criterion of GP hyperparameter training.
        - lr: float, the learning rate of Adam optimiser
        - optimiser: string, select the optimiser ["L-BFGS-B", "BoTorch", "Adam"]

    Returns:
        - model: gpytorch.models, function of GP model.
    """
    model.train()
    model.likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        if optimiser == "BoTorch":
            mll = fit_gpytorch_mll(mll)
        elif optimiser == "L-BFGS-B":
            mll = train_GP_with_BFGS(mll, training_iter, thresh)

        elif optimiser == "Adam":
            mll = train_GP_with_Adam(mll, lr, training_iter, thresh)
        else:
            raise Exception("The given optimiser is not defined")
    except:
        warnings.warn("Optimiser " + optimiser + " failed. Optimising again with Adam...")
        mll = train_GP_with_Adam(mll, lr, training_iter, thresh)
    return model


def update_gp(train_x, train_y, gp_kernel, device, lik=1e-10, training_iter=50, thresh=0.01, lr=0.1, rng=10, train_lik=False, optimiser="BoTorch"):
    """
    Input:
        - train_x: torch.tensor, inputs. torch.Size(n_data, n_dims)
        - train_y: torch.tensor, observations
        - gp_kernel: gpytorch.kernels, GP kernel model
        - device: torch.device, cpu or cuda
        - lik: float, the initial value of GP likelihood noise variance
        - train_iter: int, the maximum iteration for GP hyperparameter training.
        - thresh: float, the threshold as a stopping criterion of GP hyperparameter training.
        - lr: float, the learning rate of Adam optimiser
        - rng: int, tne range coefficient of GP likelihood noise variance
        - train_like: bool, flag whether or not to update GP likelihood noise variance
        - optimiser: string, select the optimiser ["L-BFGS-B", "BoTorch", "Adam"]

    Output:
        - model: gpytorch.models, function of GP model.
    """
    model = set_gp(train_x, train_y, gp_kernel, device, lik=lik, rng=rng, train_lik=train_lik)
    model = train_GP(model, training_iter=training_iter, thresh=thresh, lr=lr, optimiser=optimiser)
    return model


def predict(test_x, model):
    """
    Fast variance inference is made with LOVE via fast_pred_var().
    For accurate variance inference, you can just comment out the part.

    Input:
        - model: gpytorch.models, function of GP model.

    Output:
        - pred.mean; torch.tensor, the predictive mean
        - pred.variance; torch.tensor, the predictive variance
    """
    model.eval()
    model.likelihood.eval()

    with torch.no_grad():
        try:
            with gpytorch.settings.fast_pred_var():
                pred = model.likelihood(model(test_x))
        except:
            try:
                pred = model.likelihood(model(test_x))
            except:
                warnings.warn("Cholesky failed. Adding more jitter...")
                with gpytorch.settings.cholesky_jitter(float_value=1e-2):
                    pred = model.likelihood(model(test_x))
    return pred.mean, pred.variance

def predict_mean(test_x, model):
    """
    Fast variance inference is made with LOVE via fast_pred_var().
    For accurate variance inference, you can just comment out the part.

    Input:
        - model: gpytorch.models, function of GP model.

    Output:
        - pred.mean; torch.tensor, the predictive mean
        - pred.variance; torch.tensor, the predictive variance
    """
    pred_mean, _ = predict(test_x, model)
    return pred_mean

def get_cov_cache(model):
    """
    woodbury_inv = K(Xobs, Xobs)^(-1)
    S @ S.T = woodbury_inv

    Input:
        - model: gpytorch.models, function of GP model, typically self.wsabi.model in _basq.py

    Output:
        - woodbury_inv: torch.tensor, the inverse of Gram matrix K(Xobs, Xobs)^(-1)
        - Xobs: torch.tensor, the observed inputs X
        - lik_var: torch.tensor, the GP likelihood noise variance
    """
    Xobs = model.train_inputs[0]
    lik_var = model.likelihood.noise
    try:
        S = model.prediction_strategy.covar_cache
    except:
        model.eval()
        mean = Xobs[0].unsqueeze(0)
        model(mean)
        S = model.prediction_strategy.covar_cache
    woodbury_inv = S @ S.T
    return woodbury_inv, Xobs, lik_var


def predictive_covariance(x, y, model):
    """
    Input:
        - x: torch.tensor, inputs x
        - y: torch.tensor, inputs y
        - model: gpytorch.models, function of GP model.

    Output:
        - cov_xy: torch.tensor, predictive covariance matrix
    """
    woodbury_inv, Xobs, lik_var = get_cov_cache(model)
    Kxy = model.covar_module.forward(x, y)
    KxX = model.covar_module.forward(x, Xobs)
    KXy = model.covar_module.forward(Xobs, y)
    cov_xy = Kxy - KxX @ woodbury_inv @ KXy

    """
    if len(x.shape) == 3 or len(y.shape) == 3:
        d = min(x.shape[1], y.shape[1])
        cov_xy[:, range(d), range(d)] += lik_var        
    else:
        d = min(len(x), len(y))
        cov_xy[range(d), range(d)] += lik_var
    """
    return cov_xy
