import time
import copy
import torch
import warnings
from ._sampler import EmpiricalSampler
from ._kernel import Kernel
from ._pi import PI, PI_FBGP

class Sober(EmpiricalSampler):
    def __init__(
        self,
        prior,
        model,
        eps=0,
        thresh=5,
        sampler_type="lfi",
    ):
        """
        Batch Bayesian optimisation as batch Bayesian quadrature
        
        Args:
        - prior: class, the class of prior distribution
        - model: gpytorch.models, function of GP model.
        - eps: float, the machine epsilon (the smallest number of floating point).
               For double precision; eps = torch.finfo().min
        - thresh: int, the number of non-zero weights which regrads anomalies.
        - sampler_type: string, Select from "lfi" or "ts". LFI = likelihood-free inference, TS = Thompson sampling
        """
        self.sampler_type = sampler_type
        self.check_model_type(model)
        pi, kernel = self.initialisation(model)
        self.n_batches_until_reset = 3
        super().__init__(prior, pi, kernel, eps=eps, thresh=thresh, label=prior.type)  # EmpiricalSampler class initialisation
    
    def check_model_type(self, model):
        # check fully Bayesian GP model or not
        if hasattr(model, "is_fbgp"):
            self.fbgp = True
            self.is_bq = False
            self.n_init = len(model.fobs)
        elif hasattr(model, "is_bq"):
            self.fbgp = False
            self.is_bq = True
            self.n_init = len(model.fobs)
        else:
            self.fbgp = False
            self.is_bq = False
            self.n_init = len(model.train_targets)            
        
    def initialisation(self, model):
        """
        Set pi and kernel
        
        Args:
        - model: gpytorch.models, function of GP model.
        """
        if not self.fbgp:
            pi = PI(model, label=self.sampler_type)
            kernel = Kernel(model)
        else:
            pi = PI_FBGP(model)
            kernel = model.marginal_predictive_covariance
        return pi, kernel
    
    def update_model(self, model):
        """
        Set pi and kernel
        
        Args:
        - model: gpytorch.models, function of GP model.
        """
        pi, kernel = self.initialisation(model)
        super().__init__(self.prior, pi, kernel, eps=self.eps, thresh=self.thresh, label=self.prior.type)
    
    def should_reset_prior(self, batch_size, recycle_prior):
        """
        Check whether or not the prior should reset
        
        Args:
        - batch_size: int, the number of batch samples
        - recycle_prior: bool, recycle the previous prior if true, otherwise not.
        
        Return:
        - flag: bool, the prior should reset if true, otherwise not.
        """
        if not self.fbgp:
            targets = self.pi.model.train_targets
        else:
            targets = self.pi.model.fobs
        
        n_targets = len(targets)
        y_max = targets.max()
        cummax = targets.cummax(0).values
        learning_length = n_targets - self.n_init
        if (learning_length == 0) or (learning_length == batch_size):
            return False
        
        idx_max = torch.where((cummax >= y_max).diff() == True)[0][0]
        n_interations = torch.tensor(learning_length / batch_size).ceil().long()
        for n_batches in range(1, n_interations+1):
            idx_batch = n_batches * batch_size
            if idx_batch >= idx_max:
                break
        n_nonimproved_batches = n_interations - n_batches + 2
        if n_nonimproved_batches >= self.n_batches_until_reset:
            return True
        elif not recycle_prior:
            return True
        else:
            return False
    
    def next_batch(
        self, 
        n_rec, 
        n_nys,
        batch_size, 
        calc_obj=None, 
        return_weights=False,
        recycle_prior=True,
        verbose=False,
    ):
        """
        Sampling the next batch location via kernel recombination.
        
        Args:
        - n_rec: int, the number of samples for recombination
        - n_nys: int, the number of samples for Nyström approximation
        - batch_size: int, the number of batch samples
        - calc_obj: class, the acquisition function (AF). Do not use AF if None.
        - return_weights: bool, return quadrature weights if true, otherwise not.
        - recycle_prior: bool, recycle the previous prior if true, otherwise not.
        - verbose: bool, show progress if truem otherwise not.
        
        Return:
        - X_batch: torch.tensor, the next batch samples
        """
        if verbose:
            start = time.monotonic()
            print("--- generating the candidates from pi...")
        if not self.label == "dataset":
            if self.should_reset_prior(batch_size, recycle_prior):
                print("The prior was initialised.")
                self.initialise_prior()
            X_cand, X_nys, weights = self.sampling_candidates(n_rec, n_nys, verbose=verbose)
        else:
            X_cand, X_nys, weights = self.sampling_datasets(n_rec, n_nys)
            
        if verbose:
            intermidiate = time.monotonic()
            print(f"--- Finished {intermidiate - start:.3e} [s]")
            print("|| summary of sampling ||")
            print(f" # of recombination samples: {len(X_cand):.3e}")
            print(f" # of Nyström samples: {len(X_nys):.3e}")
            print(f" # of nonzero weights: {(weights > 0).sum():.3e}")
            print("--- Start kernel recombination...")
        idx_rchq, w_rchq = self.sampling_recombination(
            X_cand,
            X_nys,
            weights,
            batch_size,
            calc_obj=calc_obj,
        )
        X_batch = X_cand[idx_rchq]
        if verbose:
            end = time.monotonic()
            print(f"--- Finished all tasks {end - start:.3e} [s]")
            
        if return_weights:
            return w_rchq, X_batch
        else:
            return X_batch