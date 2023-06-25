import copy
import torch
import warnings
from ._prior_update import update_mixed_prior, update_binary_prior, update_categorical_prior
from ._weights import WeightsStabiliser
from ._rchq import recombination
from ._wkde import WeightedKernelDensityEstimation


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
        self.thresh_initial = copy.deepcopy(thresh)
        self.prior = prior
        self.pi = pi
        self.label = label
        self.flag = False
        
    def initialise_prior(self):
        """
        Initialise prior
        """
        self.prior = copy.deepcopy(self.prior_initial)

    def update_prior(self, X_cand, weights, verbose=False):
        """
        Update prior
        """
        if self.label == "mixedbinary":
            self.prior = update_mixed_prior(X_cand, weights, self.prior, label="binary")
            if verbose:
                print("The optimised weights")
                print(self.prior.prior_binary.prior_binary.probs)
        elif self.label == "mixedcategorical":
            self.prior = update_mixed_prior(X_cand, weights, self.prior, label="categorical")
            if verbose:
                print("The optimised weights")
                print(self.prior.prior_disc.cat.probs.reshape(
                    self.prior.prior_disc.n_dims,
                    self.prior.prior_disc.n_categories,
                ))
        elif self.label == "continuous":
            self.prior = WeightedKernelDensityEstimation(
                X_cand, weights, self.prior.n_dims,
            )
        elif self.label == "categorical":
            self.prior = update_categorical_prior(
                weights, X_cand, self.prior,
            )
            if verbose:
                print("The optimised weights")
                print(self.prior.cat.probs.reshape(
                    self.prior.n_dims, self.prior.n_categories
                ))
        elif self.label == "binary":
            self.prior.prior_binary = update_binary_prior(
                weights, X_cand, self.prior.prior_binary,
            )
            if verbose:
                print("The optimised weights")
                print(self.prior.prior_binary.probs)
        else:
            raise ValueError('The domain type should be from "continuous", "binary", "categorical", "mixedbinary", "mixedcategorical"')
    
    def check_categorical(self):
        """
        Check whether or not the domain is categorical.
        
        Return:
        - flag: bool, categorical if true, otherwise not.
        """
        if self.label == "mixedcategorical":
            return True
        elif self.label == "categorical":
            return True
        else:
            return False
    
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
    
    def categorical_sampling(self, n_rec):
        """
        Sampling from prior with weights
        
        Args:
        - n_rec: int, the number of samples
        
        Return:
        - X_cand: torch.tensor, samples
        - weights: torch.tensor, weights
        """
        X_cand, X_indices = self.prior.sample_both(n_rec)
        weights = self.pi(X_cand) / self.prior.pdf(X_indices)
        weights = self.cleansing_weights(weights)
        return X_cand, X_indices, weights
    
    def recursive_sampling(self, n_rec, n_repeat=5, verbose=False):
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
        X_indices_accepted = []
        weights_accepted = []
        self.flag = False
        for i in range(n_repeat):
            if verbose:
                print(str(i)+"-th recursive sampling...")
            if self.check_categorical():
                X_cand, X_indices, weights = self.categorical_sampling(n_rec)
            else:
                X_cand, weights = self.sampling(n_rec)
            
            idx = (weights > 0)
            if not idx.sum() == 0:
                X_accepted.append(X_cand[idx])
                weights_accepted.append(weights[idx])
                n_accepted += idx.sum().item()
                if self.check_categorical():
                    X_indices_accepted.append(X_indices[idx])
            
            if (n_accepted > self.thresh):
                break
        
        if n_accepted == 0:
            if verbose:
                print("Weighted sampling unsuccessful. Uniform random sampling instad...")
            self.flag = True
            if self.check_categorical():
                X_cand, X_indices, weights = self.categorical_sampling(n_rec)
                weights = torch.ones(n_rec) / n_rec
                return X_cand, X_indices, weights
            else:
                X_cand, weights = self.sampling(n_rec)
                weights = torch.ones(n_rec) / n_rec
                return X_cand, weights
        else:
            X_cand = torch.vstack(X_accepted)
            weights = torch.cat(weights_accepted)
            weights = self.cleansing_weights(weights)
            if self.check_categorical():
                X_indices = torch.vstack(X_indices_accepted)
                return X_cand, X_indices, weights
            else:
                return X_cand, weights
    
    def sampling_candidates(self, n_rec, n_nys, verbose=False):        
        """
        Sampling from pi with weights
        
        Args:
        - n_rec: int, the number of samples for recombination
        - n_nys: int, the number of samples for Nyström approximation
        - verbose: bool, show progress if truem otherwise not.
        
        Return:
        - X_cand: torch.tensor, samples for recombination
        - X_nys: torch.tensor, samples for Nyström approximation
        - weights: torch.tensor, weights
        """
        assert n_rec > n_nys
        
        if verbose:
            print("initial sampling...")
        if self.check_categorical():
            X_cand, X_indices, weights = self.categorical_sampling(n_rec)
        else:
            X_cand, weights = self.sampling(n_rec)
        if self.check_weights(weights):
            if verbose:
                print("update prior...")
                
            if self.check_categorical():
                self.update_prior(X_indices, weights, verbose=verbose)
                self.thresh = n_nys
                X_cand, _, weights = self.recursive_sampling(n_rec, n_repeat=self.thresh, verbose=verbose)
            else:
                self.update_prior(X_cand, weights, verbose=verbose)
                self.thresh = n_nys
                X_cand, weights = self.recursive_sampling(n_rec, n_repeat=self.thresh, verbose=verbose)
        else:
            print("Failed to update prior. Trying recursive sampling...")
            if self.check_categorical():
                X_cand, X_indices, weights = self.recursive_sampling(n_rec, n_repeat=self.thresh, verbose=verbose)
                if self.flag:
                    X_nys = X_cand[:n_nys]
                    return X_cand, X_nys, weights
                self.update_prior(X_indices, weights, verbose=verbose)
                self.thresh = n_nys
                X_cand, _, weights = self.recursive_sampling(n_rec, n_repeat=self.thresh, verbose=verbose)
            else:
                X_cand, weights = self.recursive_sampling(n_rec, n_repeat=self.thresh, verbose=verbose)
                if self.flag:
                    X_nys = X_cand[:n_nys]
                    return X_cand, X_nys, weights
                self.update_prior(X_cand, weights)
                self.thresh = n_nys
                X_cand, weights = self.recursive_sampling(n_rec, n_repeat=self.thresh, verbose=verbose)
        
        idx_nys = self.weighted_resampling(weights, n_nys)
        X_nys = X_cand[idx_nys]
        self.thresh = copy.deepcopy(self.thresh_initial)
        return X_cand, X_nys, weights
    
    def sampling_datasets(self, n_rec, n_nys):        
        """
        Sampling from dataset with weights
        
        Args:
        - n_rec: int, the number of samples for recombination
        - n_nys: int, the number of samples for Nyström approximation
        
        Return:
        - X_cand: torch.tensor, samples for recombination
        - X_nys: torch.tensor, samples for Nyström approximation
        - weights: torch.tensor, weights
        """
        assert n_rec > n_nys
    
        if n_rec > self.prior.n_available:
            X_cand = self.prior.available_candidates()
        else:
            X_cand = self.prior.sample_feature(n_rec)
        
        weights = self.pi(X_cand)
        weights = self.cleansing_weights(weights)
        
        idx_nys = self.weighted_resampling(weights, n_nys)
        X_nys = X_cand[idx_nys]
        return X_cand, X_nys, weights