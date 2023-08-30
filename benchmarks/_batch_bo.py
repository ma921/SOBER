import copy
import math
import torch
from torch.quasirandom import SobolEngine
from dataclasses import dataclass
import gpytorch
from botorch.generation import MaxPosteriorSampling
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from .gp_sampling import decoupled_sampler
from .dpp_ts_bo.gp import GP
from .dpp_ts_bo.snippet_dppts import MCMC_DPP_Batched_TS_GP
from SOBER._utils import TensorManager
from SOBER._sober import Sober
tm = TensorManager()

def thompson_sampling(model, prior, n_rec, batch_size):
    X_cand = prior.sample(n_rec)
    with gpytorch.settings.max_cholesky_size(float("inf")), torch.no_grad():
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_batch = thompson_sampling(X_cand, num_samples=batch_size)
    return X_batch

def decoupled_thompson_sampling(model, prior, n_rec, batch_size):
    X_cand = prior.sample(n_rec)
    ds = decoupled_sampler(
        model=model,
        sample_shape=[batch_size],
        num_basis=4096, #1024,
    )
    y_cand = ds(X_cand).squeeze(-1).t()

    X_batch = tm.zeros(batch_size, X_cand.shape[-1])
    for i in range(batch_size):
        ind_max = y_cand[:, i].argmax()
        X_batch[i, :] = X_cand[ind_max, :]
        y_cand[ind_max, :] = -float("inf")
    return X_batch

def DPP_TS(model, TrueFunction, prior, n_rec, batch_size):
    X_cand = prior.sample(n_rec)    
    wrapped_model = GP(model)
    gp = MCMC_DPP_Batched_TS_GP(model.train_inputs[0], TrueFunction, wrapped_model)
    X_cand = prior.sample(n_rec)
    output = gp.step(X_cand, batch_size=batch_size)
    X_batch = output['x_batch']
    return X_batch

def GIBBON(model, prior, n_rec, batch_size):
    candidate_set = prior.sample(n_rec)
    qGIBBON = qLowerBoundMaxValueEntropy(model, candidate_set)
    X_batch, _ = optimize_acqf(
        acq_function=qGIBBON,
        bounds=prior.bounds,
        q=batch_size,
        num_restarts=5,
        raw_samples=batch_size,
        sequential=True,
    )
    return X_batch

def Hallucination(model, set_model, prior, batch_size):
    X_batch = []
    X_fantasy = copy.deepcopy(model.train_inputs[0])
    Y_fantasy = copy.deepcopy(model.train_targets)

    for i_b in range(batch_size):
        model = set_model(X_fantasy, Y_fantasy)
        eta = Y_fantasy.max()
        EI = ExpectedImprovement(model, eta)

        X_next, _ = optimize_acqf(
            EI,
            bounds=prior.bounds,
            q=1,
            num_restarts=5,
            raw_samples=batch_size,
        )
        pred = model.likelihood(model(X_next))

        Y_next = pred.mean.detach() * Y_fantasy.std() + Y_fantasy.mean()
        X_fantasy = torch.cat((X_fantasy, X_next), dim=0)
        Y_fantasy = torch.cat((Y_fantasy, Y_next), dim=0)
        X_batch.append(X_next)

    X_batch = torch.cat(X_batch, dim=0)
    return X_batch

def local_penalisation(
    model,  # GP model
    prior,
    batch_size,
    LIPSCHITZ=1,
):
    eta = model.train_targets.max()
    X_batch = []
    for i in range(batch_size):
        p_ei = penalised_ei(X_batch, model, eta, lipschitz=LIPSCHITZ, maximize=True)
        X_next, acq_value = optimize_acqf(
            p_ei,
            bounds=prior.bounds,
            q=1,
            num_restarts=5,
            raw_samples=batch_size,
        )
        X_batch.append(X_next)
    X_batch = torch.cat(X_batch, dim=0)
    return X_batch

def turbo(state, model, prior, batch_size):
    X = normalize(model.train_inputs[0], prior.bounds)
    Y = copy.deepcopy(model.train_targets)
    dim = X.shape[-1]
    n_candidates = min(5000, max(2000, 200 * dim))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / 1.0)) #len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=tm.dtype, device=tm.device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = (
        torch.rand(n_candidates, dim, dtype=tm.dtype, device=tm.device)
        <= prob_perturb
    )
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=tm.device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points
    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():  # We don't need gradients when using TS
        X_next = thompson_sampling(X_cand, num_samples=batch_size)
    return unnormalize(X_next, prior.bounds)

def SOBER_TS(model, prior, batch_size, n_cand_super=20000, n_cand=2000, n_nys=200):
    sober = Sober(prior, model)
    
    X_cand = decoupled_thompson_sampling(model, prior, n_cand_super, n_cand)
    weights = tm.ones(n_cand) / n_cand
    X_nys = X_cand[:n_nys]
    
    idx_rchq, w_rchq = sober.sampling_recombination(
        X_cand,
        X_nys,
        weights,
        batch_size,
        calc_obj=None,
    )
    X_batch = X_cand[idx_rchq]
    return X_batch

### Auxiliary codes

# 1. Local Penalisation
class penalised_ei(ExpectedImprovement):
    def __init__(
            self,
            X_batch,
            *args,
            lipschitz=1,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.X_batch = X_batch
        self.lipschitz = lipschitz

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ei = super().forward(X)
        for xb in self.X_batch:
            pred = self.model(xb)
            mu = pred.mean[0]
            var = pred.variance[0]
            z = (self.lipschitz * (X - xb).pow(2).sum(-1).sqrt() - self.best_f + mu) / torch.sqrt(2 * var)
            ei = 0.5 * torch.erfc(-z.squeeze()) * ei
        return ei

# 2. TurBO
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state