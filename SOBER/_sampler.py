import copy
import torch
import warnings
from ._prior import Uniform, BinaryPrior, CategoricalPrior, MixedBinaryPrior, MixedCategoricalPrior
from ._prior_update import update_mixed_prior, update_binary_prior, update_categorical_prior, update_continuous_prior
from ._weights import WeightsStabiliser
from ._rchq import recombination
from ._utils import TensorManager


class RecombinationSampler(WeightsStabiliser, TensorManager):
    def __init__(
        self,
        kernel,
        thresh=5,
    ):
        """
        Sampling via kernel recombination.
        
        Args:
        - kernel: class, the class of kernel
        """
        WeightsStabiliser.__init__(self, thresh=thresh)  # WeightsStabiliser class initialisation
        TensorManager.__init__(self) # TensorManager class initialisation
        self.kernel = kernel
        
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
            self.dtype,
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
        thresh=5,
        label="mixedbinary", 
    ):
        """
        Sampling from pi.
        
        Args:
        - prior: class, the class of prior distribution
        - pi: class, the class of pi
        - kernel: class, the class of kernel
        - label: string, prior type. Select from "continuous", "binary", "categorical", "mixedbinary", "mixedcategorical".
        """
        super().__init__(kernel, thresh=thresh)  # RecombinationSampler class initialisation
        #self.prior_initial = copy.deepcopy(prior)
        self.thresh_initial = copy.deepcopy(thresh)
        self.prior = prior
        self.pi = pi
        self.label = label
        self.flag = False
        
    def initialise_prior(self):
        """
        Initialise prior
        """
        if self.label == "continuous":
            self.prior = Uniform(self.prior.bounds)
        elif self.label == "binary":
            self.prior = BinaryPrior(self.prior.n_dims)
        elif self.label == "categorical":
            self.prior = CategoricalPrior(self.prior.categories)
        elif self.label == "mixedbinary":
            self.prior = MixedBinaryPrior(
                self.prior.n_dims_cont, 
                self.prior.n_dims_binary,
                self.prior.bounds,
                self.prior.continous_first,
            )
        elif self.label == "mixedcategorical":
            self.prior = MixedCategoricalPrior(
                self.prior.n_dims_cont, 
                self.prior.n_dims_disc,
                self.prior.categories,
                self.prior.bounds,
                self.prior.continous_first,
            )

    def update_prior(self, X_cand, weights, verbose=False):
        """
        Update prior
        
        Args:
        - X_cand: torch.tensor, samples
        - weights: torch.tensor, weights
        - verbose: bool, show progress if truem otherwise not.
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
            self.prior = update_continuous_prior(
                X_cand, weights, self.prior, self.prior.n_dims,
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
                print("Weighted sampling unsuccessful. Uniform random sampling instead...")
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
        
        if self.label == "continuous":
            X_nys = self.kmeans_resampling(X_cand, n_clusters=n_nys)
        else:
            idx_nys = self.deweighted_resampling(weights, n_nys)
            X_nys = X_cand[idx_nys]
        
        self.thresh = copy.deepcopy(self.thresh_initial)
        return X_cand, X_nys, weights
    
    def adaptive_pruning(self, weights, n_rec, n_nys, thresh=1e-3):
        """
        Pruning the candindates from dataset with weights
        
        Args:
        - weights: torch.tensor, weights
        - n_rec: int, the number of samples for recombination
        - n_nys: int, the number of samples for Nyström approximation
        
        Return:
        - idx_sampled: torch.tensor, indices where X_cand is sampled
        """
        indices = weights.argsort(descending=True)
        try:
            n_accepted = torch.where(weights[indices] > thresh)[0][-1] + 1
            if n_accepted >= n_rec:
                n_pruned = n_rec
            elif n_nys >= n_accepted:
                n_pruned = n_nys
            else:
                n_pruned = n_accepted
        except:
            n_pruned = n_nys
        idx_sampled = indices[:n_pruned]
        return idx_sampled
    
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
        - idx_sampled: (optional) torch.tensor, indices where X_cand is sampled
        """
        assert n_rec > n_nys
    
        X_cand = self.prior.available_candidates()
        weights = self.pi(X_cand)
        
        if self.dataset_pruning:
            idx_sampled = self.adaptive_pruning(weights, n_rec, n_nys)
            X_cand = X_cand[idx_sampled]
            weights = weights[idx_sampled]
        
        weights = self.cleansing_weights(weights)
        idx_nys = self.deweighted_resampling(weights, n_nys)
        X_nys = X_cand[idx_nys]
        
        if self.dataset_pruning:
            return idx_sampled, X_cand, X_nys, weights
        else:
            return X_cand, X_nys, weights

class MixtureSampler:
    def __init__(self, prior, sober, ratio_wkde=0.5):
        """
        Sampling from posterior.
        
        Args:
        - prior: class, the class of prior distribution
        - sober: class, the class of the learnt Sober
        - ratio_wkde: float, the proportion to sample from pi
        """
        self.prior = prior
        self.sober = sober
        self.bounds = prior.bounds
        self.ratio_wkde = ratio_wkde
    
    def sample(self, n_samples):
        """
        Sampling from the mixture of prior and pi
        
        Args:
           - n_samples: int, number of samples to draw
           
        Returns:
            - samples: torch.tensor, the samples from mixture density
        """
        n_wkde = int(self.ratio_wkde * n_samples)
        n_prior = int((1-self.ratio_wkde) * n_samples)

        tm = TensorManager()
        
        if n_wkde:
            samples_wkde = self.sober.prior.sample(n_wkde)
        else:
            samples_wkde = torch.zeros(
                [0, self.sober.prior.n_dims], dtype=tm.dtype, device=tm.device
            )
        if n_prior:
            samples_prior = self.prior.sample(n_prior)
        else:
            samples_prior = torch.zeros(
                [0, self.sober.prior.n_dims], dtype=tm.dtype, device=tm.device
            )
        samples = torch.vstack([samples_wkde, samples_prior])
        return samples
    
    def pdf(self, X):
        """
        Probability density function of the mixture distribution
        
        Args:
           - n_samples: int, number of samples to draw
           
        Returns:
            - samples: torch.tensor, the samples from mixture density
        """
        pdfs = self.sober.prior.pdf(X)/2 + self.prior.pdf(X)/2
        return pdfs
