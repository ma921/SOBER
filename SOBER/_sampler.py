import copy
import torch
import warnings
from ._prior_update import update_mixed_prior
from ._weights import WeightsStabiliser
from ._rchq import recombination


class RecombinationSampler(WeightsStabiliser):
    def __init__(
        self,
        kernel,
        eps=0,
        thresh=5,
    ):
        """
        Sampling via kernel recombination.
        
        Args:
        - kernel: class, the class of kernel
        - eps: float, the machine epsilon (the smallest number of floating point).
               For double precision; eps = torch.finfo().min
        - thresh: int, the number of non-zero weights which regrads anomalies.
        """
        super().__init__(eps=eps, thresh=thresh)  # WeightsStabiliser class initialisation
        self.kernel = kernel
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    def sampling_recombination(
        self,
        X_cand,
        X_nys,
        weights,
        batch_size,
        calc_obj=None,
    ):
        """
        Sampling via kernel recombination.
        
        Args:
        - X_cand: torch.tensor, samples for recombination
        - X_nys: torch.tensor, samples for Nyström approximation
        - weights: torch.tensor, weights
        - batch_size: int, the number of batch samples
        - calc_obj: class, the acquisition function.
        
        Return:
        - idx_rchq: torch.tensor, the indices selected for the next batch
        - w_rchq: torch.tensor, the quadrature weights
        """
        idx_rchq, w_rchq = recombination(
            X_cand,
            X_nys,
            batch_size,
            self.kernel,
            self.device,
            init_weights=weights,
            calc_obj=calc_obj,
        )
        return idx_rchq, w_rchq

class EmpiricalSampler(RecombinationSampler):
    def __init__(
        self,
        prior,
        pi,
        kernel,
        eps=0,
        thresh=5,
        label="mixedbinary", 
    ):
        """
        Sampling from pi.
        
        Args:
        - prior: class, the class of prior distribution
        - pi: class, the class of pi
        - kernel: class, the class of kernel
        - eps: float, the machine epsilon (the smallest number of floating point).
               For double precision; eps = torch.finfo().min
        - thresh: int, the number of non-zero weights which regrads anomalies.
        - label: string, prior type. Select from "continuous", "binary", "categorical", "mixedbinary", "mixedcategorical".
        """
        super().__init__(kernel, eps=eps, thresh=thresh)  # RecombinationSampler class initialisation
        self.prior_initial = copy.deepcopy(prior)
        self.prior = prior
        self.pi = pi
        self.label = label

    def update_prior(self, X_cand, weights):
        """
        Update prior
        """
        if self.label == "mixedbinary":
            self.prior = update_mixed_prior(X_cand, weights, self.prior, label="binary")
        else:
            raise ValueError('The domain type should be from "continuous", "binary", "categorical", "mixedbinary", "mixedcategorical"')
    
    def sampling(self, n_rec):
        """
        Sampling from prior with weights
        
        Args:
        - n_rec: int, the number of samples
        
        Return:
        - X_cand: torch.tensor, samples
        - weights: torch.tensor, weights
        """
        X_cand = self.prior.sample(n_rec)
        weights = self.pi(X_cand) / self.prior.pdf(X_cand)
        weights = self.cleansing_weights(weights)
        return X_cand, weights
    
    def recursive_sampling(self, n_rec, n_repeat=5):
        """
        Sampling from prior with weights recursively
        
        Args:
        - n_rec: int, the number of samples
        - n_repeat: int, the number of iterations
        
        Return:
        - X_cand: torch.tensor, samples
        - weights: torch.tensor, weights
        """
        n_accepted = 0
        X_accepted = []
        weights_accepted = []
        for i in range(n_repeat):
            X_cand, weights = self.sampling(n_rec)
            idx = (weights > 0)
            if not idx.sum() == 0:
                X_accepted.append(X_cand[idx])
                weights_accepted.append(weights[idx])
                n_accepted += idx.sum().item()
            
            #print(str(i)+"-th iterations: "+str(n_accepted))
            if (n_accepted > self.thresh):
                break

        X_cand = torch.vstack(X_accepted)
        weights = torch.cat(weights_accepted)
        weights = self.cleansing_weights(weights)
        return X_cand, weights
    
    def sampling_candidates(self, n_rec, n_nys):        
        """
        Sampling from pi with weights
        
        Args:
        - n_rec: int, the number of samples for recombination
        - n_nys: int, the number of samples for Nyström approximation
        
        Return:
        - X_cand: torch.tensor, samples for recombination
        - X_nys: torch.tensor, samples for Nyström approximation
        - weights: torch.tensor, weights
        """
        assert n_rec > n_nys
        
        X_cand, weights = self.sampling(n_rec)
        if self.check_weights(weights):
            self.update_prior(X_cand, weights)
            #X_cand, weights = self.sampling(n_rec)
            self.thresh = n_nys
            X_cand, weights = self.recursive_sampling(n_rec, n_repeat=self.thresh)
        else:
            #warnings("Failed to update prior. Trying recursive sampling")
            print("Failed to update prior. Trying recursive sampling")
            X_cand, weights = self.recursive_sampling(n_rec, n_repeat=self.thresh)
            self.update_prior(X_cand, weights)
            self.thresh = n_nys
            X_cand, weights = self.recursive_sampling(n_rec, n_repeat=self.thresh)
        
        idx_nys = self.weighted_resampling(weights, n_nys)
        X_nys = X_cand[idx_nys]
        return X_cand, X_nys, weights
    
    def next_batch(self, n_rec, n_nys, batch_size, calc_obj=None):
        """
        Sampling the next batch location via kernel recombination.
        
        Args:
        - n_rec: int, the number of samples for recombination
        - n_nys: int, the number of samples for Nyström approximation
        - batch_size: int, the number of batch samples
        - calc_obj: class, the acquisition function.
        
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
        