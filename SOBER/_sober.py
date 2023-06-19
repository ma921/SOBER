import copy
import torch
import warnings
from ._sampler import EmpiricalSampler
from ._kernel import Kernel
from ._pi import PI

class Sober(EmpiricalSampler):
    def __init__(
        self,
        prior,
        model,
        eps=0,
        thresh=5,
        domain_type="mixedbinary",
        sampler_type="lfi",
    ):
        """
        Sampling from pi.
        
        Args:
        - prior: class, the class of prior distribution
        - model: gpytorch.models, function of GP model.
        - eps: float, the machine epsilon (the smallest number of floating point).
               For double precision; eps = torch.finfo().min
        - thresh: int, the number of non-zero weights which regrads anomalies.
        - domain_type: string, prior type. Select from "continuous", "binary", "categorical", "mixedbinary", "mixedcategorical".
        - sampler_type: string, Select from "lfi" or "ts". LFI = likelihood-free inference, TS = Thompson sampling
        """
        pi, kernel = self.initialisation(model, sampler_type=sampler_type)
        super().__init__(prior, pi, kernel, eps=eps, thresh=thresh, label=domain_type)  # EmpiricalSampler class initialisation
        
    def initialisation(self, model, sampler_type="lfi"):
        """
        Set pi and kernel
        """
        pi = PI(model, label=sampler_type)
        kernel = Kernel(model)
        return pi, kernel
    
    def __call__(self, n_rec, n_nys, batch_size, calc_obj=None):
        """
        Sampling the next batch location via kernel recombination.
        
        Args:
        - n_rec: int, the number of samples for recombination
        - n_nys: int, the number of samples for Nyström approximation
        - batch_size: int, the number of batch samples
        - calc_obj: class, the acquisition function (AF). Do not use AF if None.
        
        Return:
        - X_batch: torch.tensor, the next batch samples
        """
        X_cand, X_nys, weights = self.sampling_candidates(n_rec, n_nys)
        idx_rchq, w_rchq = self.sampling_recombination(
            X_cand,
            X_nys,
            weights,
            batch_size,
            calc_obj=calc_obj,
        )
        X_batch = X_cand[idx_rchq]
        return X_batch
        