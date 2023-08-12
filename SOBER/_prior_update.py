import torch
import torch.optim as optim
import torch.distributions as D
from abc import ABC, abstractmethod
from ._wkde import WeightedKernelDensityEstimation
from ._utils import TensorManager


class BaseMLE(ABC, TensorManager):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def objective(self, X):
        r"""Optimization objective (weights)"""
        pass
    
    @abstractmethod
    def closure(self, X):
        r"""Optimization loop"""
        pass
    
    @abstractmethod
    def run(self, X):
        r"""Running optimization loop"""
        pass
    
    @abstractmethod
    def update_prior(self, X):
        r"""Return the updated prior"""
        pass

class BernoulliMLE(BaseMLE):
    def __init__(self, weights, x_binary, n_max=5):
        """
        Update Bernoulli prior via maximum likelihood estimation (MLE).
        
        Args:
        - weights: torch.tensor, the weights at the observed input
        - x_binary: torch.tensor, the observed input
        - n_max: int, the number of L-BFGS-B iteration
        """
        super().__init__() # call TensorManager
        self.weights = weights.detach()
        self.x_binary = x_binary
        self.n_dims_binary = x_binary.size(1)
        self.n_max = n_max
    
    def objective(self, w):
        """
        The objective of L-BFGS-B loop (maximum likelihood)
        
        Args:
        - w: torch.tensor, the weights to be optimised
        
        Return:
        - ans: torch.float, the negative log likelihood of the given w
        """
        dist = D.Bernoulli(w)
        ans = self.weights @ dist.log_prob(self.x_binary).sum(axis=1)
        return -1 * ans
    
    def transform(self, w):
        """
        Sigmoid transform to make w to be bounded from 0 to 1
        
        Args:
        - w: torch.tensor, the weights to be optimised
        
        Return:
        - w_trans: torch.tensor, the transformed weights
        """
        return 1/(1 + w.exp())

    def closure(self):
        """
        A single step closure of iteration loop
        
        Return:
        objective: torch.float, the negative log likelihood of the given w
        """
        self.lbfgs.zero_grad()
        params = self.transform(self.x_lbfgs)
        objective = self.objective(params)
        objective.backward()
        return objective
    
    def run(self):
        """
        Maximum likelihood estimation of optimal weights for the Bernoulli sampler
        
        Return:
        result: torch.tensor, the optimised weights for the Bernoulli sampler
        """
        self.x_lbfgs = self.ones(self.n_dims_binary) * 0.5
        self.x_lbfgs.requires_grad = True
        
        self.lbfgs = optim.LBFGS(
            [self.x_lbfgs],
            history_size=10, 
            max_iter=4, 
            line_search_fn="strong_wolfe",
        )
        
        for i in range(self.n_max):
            self.lbfgs.step(self.closure)
        result = self.transform(self.x_lbfgs).detach()
        return result
    
    def update_prior(self, prior_binary):
        """
        Update the Bernoulli prior
        
        Args:
        - prior_binary: torch.distributions.Bernoulli, the Bernoulli prior
        
        Return:
        - prior_binary: torch.distributions.Bernoulli, the Bernoulli prior
        """
        weights_updated = self.run()
        prior_binary.probs = weights_updated
        return prior_binary
    
class CategoricalMLE(BaseMLE):
    def __init__(self, weights, x_disc, prior, n_max=5):
        """
        Update Bernoulli prior via maximum likelihood estimation (MLE).
        
        Args:
        - weights: torch.tensor, the weights at the observed input
        - x_disc: torch.tensor, the observed input
        - prior: class, the function of categorical prior
        - n_max: int, the number of L-BFGS-B iteration
        """
        super().__init__() # call TensorManager
        self.weights = weights.detach()
        self.x_disc = x_disc.detach()
        self.n_dims_disc = x_disc.size(1)
        self.prior = prior
        self.n_max = n_max
        
    def reshape_weights(self, _w):
        """
        Reshape the weights for CategoricalPrior class
        
        Args:
        - _w: torch.tensor, the one-dimensional weights to be optimised
        
        Return:
        - w: list, the reshaped weights _w
        """
        sections = torch.cat([torch.tensor([0]), self.prior.n_categories.cumsum(0)])
        w = [_w[sections[idx]:sections[idx+1]] for idx in range(self.prior.n_dims)]
        return w
    
    def objective(self, _w):
        """
        The objective of L-BFGS-B loop (maximum likelihood)
        
        Args:
        - w: torch.tensor, the weights to be optimised
        
        Return:
        - ans: torch.float, the negative log likelihood of the given w
        """
        w = self.reshape_weights(_w)
        self.prior.weights = w
        self.prior.initialise()
        ans = self.weights @ self.prior.logpdf(self.x_disc)
        return -1 * ans
    
    def transform(self, w):
        """
        Sigmoid transform to make w to be bounded from 0 to 1
        
        Args:
        - w: torch.tensor, the weights to be optimised
        
        Return:
        - w_trans: torch.tensor, the transformed weights
        """
        return 1/(1 + w.exp())
    
    def closure(self):
        """
        A single step closure of iteration loop
        
        Return:
        objective: torch.float, the negative log likelihood of the given w
        """
        self.lbfgs.zero_grad()
        params = self.transform(self.x_lbfgs)
        objective = self.objective(params)
        objective.backward(retain_graph=True)
        return objective
    
    def run(self):
        """
        Maximum likelihood estimation of optimal weights for the categorical sampler
        
        Return:
        result: torch.tensor, the optimised weights for the Bernoulli sampler
        """
        self.x_lbfgs = torch.hstack(self.prior.weights)
        self.x_lbfgs.requires_grad = True
        
        self.lbfgs = optim.LBFGS([self.x_lbfgs],
                    history_size=10,
                    max_iter=4, 
                    line_search_fn="strong_wolfe")
                    
        for i in range(self.n_max):
            self.lbfgs.step(self.closure)
        result = self.transform(self.x_lbfgs).detach()
        return result
    
    def update_prior(self, prior_disc):
        """
        Update the categorical prior
        
        Args:
        - prior_disc: class, the categorical prior
        
        Return:
        - prior_disc: class, the optimised categorical prior
        """
        weights_updated = self.run()
        prior_disc.weights = self.reshape_weights(weights_updated)
        return prior_disc
    
def update_binary_prior(weights, x_binary, prior_binary):
    """
    Update the Bernoulli prior

    Args:
    - weights: torch.tensor, the weghts at X_cand
    - X_binary: torch.tensor, the binary input
    - prior_binary: torch.distributions.Bernoulli, the Bernoulli prior

    Return:
    - prior_binary: torch.distributions.Bernoulli, the Bernoulli prior
    """
    mle_binary = BernoulliMLE(weights, x_binary)
    prior_binary = mle_binary.update_prior(prior_binary)
    return prior_binary

def update_categorical_prior(weights, x_disc, prior_categorical):
    """
    Update the categorical prior

    Args:
    - weights: torch.tensor, the weghts at X_cand
    - X_disc: torch.tensor, the categorical input
    - prior_categorical: torch.distributions.Categorical, the Categorical prior

    Return:
    - prior_categorical: torch.distributions.Categorical, the optimised Categorical prior
    """
    mle_disc = CategoricalMLE(weights, x_disc, prior_categorical)
    prior_categorical =  mle_disc.update_prior(prior_categorical)
    return prior_categorical

def update_continuous_prior(X_cand, weights, prior, n_dims):
    """
    Update the continuous prior

    Args:
    - X_cand: torch.tensor, the mixed input
    - weights: torch.tensor, the weghts at X_cand
    - prior: class, the continuous prior
    - n_dims: int, the number of dimensions

    Return:
    - prior: class, the Gaussian mixture prior
    """
    if hasattr(prior, "bounds"):
        bounds = prior.bounds
    else:
        bounds = None
        
    prior = WeightedKernelDensityEstimation(
        X_cand, weights, n_dims, bounds=bounds,
    )
    return prior

def update_mixed_prior(X_cand, weights, prior, label="binary"):
    """
    Update the mixed prior

    Args:
    - X_cand: torch.tensor, the mixed input
    - weights: torch.tensor, the weghts at X_cand
    - prior: class, the mixed prior
    - label: string, "binary" or "categorical"

    Return:
    - prior: class, the mixed prior
    """
    x_cont, x_disc = prior.separate_samples(X_cand)
    if label == "binary":
        prior.prior_binary.prior_binary = update_binary_prior(
            weights, x_disc, prior.prior_binary.prior_binary,
        )
    elif label == "categorical":
        prior.prior_disc = update_categorical_prior(
            weights, x_disc, prior.prior_disc,
        )
    else:
        raise ValueError("label should be either 'binary' or 'categorical'.")
    
    prior.prior_cont = update_continuous_prior(x_cont, weights, prior, prior.n_dims_cont)
    return prior

