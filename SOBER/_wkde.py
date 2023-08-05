import copy
import torch
import warnings
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from ._utils import SafeTensorOperator
from ._weights import WeightsStabiliser
from ._prior import BasePrior
from .mvnorm import multivariate_normal_cdf as Phi

class WeightedKernelDensityEstimation(
    WeightsStabiliser, 
    SafeTensorOperator, 
    BasePrior,
):
    def __init__(
        self,
        X,
        W,
        n_dims,
        bounds=None,
        n_kde=4096,
        bw_method='scott',
        compute_cdf=False,
    ):
        """
        Class of weighted kernel density estimation with Gaussian kernel.
        
        Args:
        - X: torch.tensor, the observed data X
        - W: torch.tensor, the weights over X
        - n_dims: int, the number of dimensions
        - bounds: torch.tensor, the lower and upper bounds for each dimension. If none, the bounds are ignored.
        - n_kde: int, the number of Gaussians for KDE.
        - bw_method: string, 'scott' or 'silverman'
        - compute_cdf: bool, compute normalised PDF of truncated multivariate normal distributions if true, otherwise not.
                             The default is false as it produces significant overhead.
        """
        WeightsStabiliser.__init__(self, eps=0, thresh=n_kde) # inherit WeightsStabiliser class
        SafeTensorOperator.__init__(self) # inherit SafeTensorOperator class
        BasePrior.__init__(self) # inherit BasePrior class
        self.n_dims = n_dims
        self.bounds = bounds
        self.n_kde_init = min([n_kde, len(X)])
        self.bw_method = bw_method
        self.compute_cdf = compute_cdf
        self.type = "continuous"
        self.initialisation(X, W)
        
    def initialise_n_kde(self):
        self.n_kde = copy.deepcopy(self.n_kde_init)
    
    def initialisation(self, X, Y):
        """
        Estimate hyperparameters and sparse Gaussians
        
        Args:
        - X: torch.tensor, the observed data X
        - Y: torch.tensor, the unnormalised weights
        """
        self.initialise_n_kde()
        if self.check_weights(Y):
            idx_accept = self.deweighted_resampling(Y, self.n_kde)
        else:
            idx_accept = self.arange(Y.size(0))[self.cleansing_weights(Y) > 0]
            self.n_kde = len(idx_accept)
            if self.n_kde < 1:
                raise ValueError("Invalid weights")
            elif self.n_kde > self.n_kde_init:
                self.initialise_n_kde()
                idx_accept = self.deweighted_resampling(Y, self.n_kde)
        
        self.Xobs = X[idx_accept]
        self.weights = self.cleansing_weights(Y[idx_accept])
        self.n_kde = self.Xobs.size(0)
        self.set_bandwidth()
        self._compute_covariance()
        if self.compute_cdf:
            self._compute_constant()
        
    def _compute_constant(self):
        p_lb = Phi(self.bounds[0], loc=self.Xobs, covariance_matrix=self.covariance)
        p_ub = Phi(self.bounds[1], loc=self.Xobs, covariance_matrix=self.covariance)
        self.constant = self.standardise_tensor(p_ub - p_lb)

    def set_bandwidth(self):
        """
        Set bandwidth (self.bw) by following the methods (self.bw_method)
        """
        self.neff = 1.0 / (self.weights ** 2).sum()
        if self.bw_method == 'scott':
            self.bw = self.neff.pow(-1./(self.n_dims+4))
        elif self.bw_method == 'silverman':
            self.bw = (self.neff * (self.n_dims+2.0)/4.0).pow(-1./(self.n_dims+4))
        
    def _compute_covariance(self):
        """
        Set covariance matrix (self.covariance)
        """
        # Compute the mean and residuals
        _mean = self.weights @ self.Xobs
        _residual = self.Xobs - _mean.unsqueeze(0)
        # Compute the biased covariance
        self._data_covariance = (_residual.T * self.weights.unsqueeze(0)) @ _residual
        # Correct for bias(http://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance)
        self._data_covariance /= 1 - self.weights.pow(2).sum()
        self.covariance = self.make_cov_psd(self._data_covariance * self.bw.pow(2))

    def pdf(self, X):
        """
        Compute the probability density function (PDF) at X
        
        Args:
        - X: torch.tensor, the observed data X
        
        
        Return:
        - Ypreds: torch.tensor, the PDF at X
        """
        n_X = len(X)

        x_AA = (
            self.Xobs.repeat(n_X, 1, 1) - X.unsqueeze(1)
        ).reshape(int(self.n_kde*n_X), self.n_dims)

        Npdfs_AA = self.safe_mvn_prob(
            self.zeros(self.n_dims),
            self.covariance,
            x_AA,
        ).reshape(n_X, self.n_kde)
        
        if not self.bounds == None:
            # Thresholding out-of-bound samples
            indices_min = (X < self.bounds[0]).any(axis=1)
            indices_max = (X > self.bounds[1]).any(axis=1)
            Npdfs_AA[indices_min] = self.zeros(self.n_kde)
            Npdfs_AA[indices_max] = self.zeros(self.n_kde)
            
            if self.compute_cdf:
                Ypreds = (self.weights / self.constant) @ Npdfs_AA.T
            else:
                Ypreds = self.weights @ Npdfs_AA.T
        else:
            Ypreds = self.weights @ Npdfs_AA.T
        return Ypreds
    
    def logpdf(self, X):
        """
        Compute the log probability density function (PDF) at X
        
        Args:
        - X: torch.tensor, the observed data X
        
        Return:
        - Ypreds: torch.tensor, the log PDF at X
        """
        return self.pdf(X).log()
    
    def rejection_sampling(self, mean, cov, cnt, n_repeat=10):
        """
        Rejection sampling from truncated Gaussian prior
        
        Args:
        - mean: torch.tensor, the mean vector of Gaussian
        - cov: torch.tensor, the covariance matrix of Gaussian
        - cnt: int, the number of samples
        - n_repeat: int, the number of iteration until len(samples) >= cnt
        
        Return:
        - samples: torch.tensor, the accepted samples from truncated Gaussian prior
        """
        samples = self.null()
        for i in range(n_repeat):
            cov = self.make_cov_psd(cov)
            samples_raw = MultivariateNormal(
                mean,
                cov,
            ).sample(torch.Size([int(n_repeat*cnt)]))
            
            indices_min = (samples_raw < self.bounds[0]).any(axis=1)
            indices_max = (samples_raw > self.bounds[1]).any(axis=1)
            indices_accept = torch.logical_or(indices_min, indices_max).logical_not()
            samples_accepted = samples_raw[indices_accept]
            samples = torch.cat([samples, samples_accepted])
            if (len(samples) >= cnt):
                break
        if (len(samples) > cnt):
            samples = samples[:cnt]
        return samples
    
    def sample_from_Gaussian(self, mean, cov, cnt, n_repeat=10):
        """
        Sampling from each Gaussian
        
        Args:
        - mean: torch.tensor, the mean vector of Gaussian
        - cov: torch.tensor, the covariance matrix of Gaussian
        - cnt: int, the number of samples
        - n_repeat: int, the number of iteration until len(samples) >= cnt
        
        Return
        - samples: torh.tensor, samples from Gaussian
        """
        if cnt == 0:
            warnings.warn("invalid Gaussian in the kernel density estimation")
            return self.null()
        elif (cov == 0).all():
            warnings.warn("invalid Gaussian in the kernel density estimation")
            return self.null()
        else:
            cov = self.make_cov_psd(cov)
            if self.bounds == None:
                samples = MultivariateNormal(
                    mean,
                    cov,
                ).sample(torch.Size([cnt]))
            else:
                samples = self.rejection_sampling(mean, cov, cnt, n_repeat=n_repeat)
            return samples
    
    def sample(self, N_rec):
        """
        Sampling from KDE
        
        Args:
        - N_rec: int, the number of samples
        
        Return
        - samples: torh.tensor, samples from KDE
        """
        cnts = self.weights * N_rec
        cnt_kde = cnts.type(torch.int)
        if cnt_kde.sum() < N_rec:
            cnt_kde = (2 * cnts).type(torch.int)

        samples = torch.cat([
            self.sample_from_Gaussian(
                self.Xobs[i],
                self.covariance,
                cnt,
            )
            for i, cnt in enumerate(cnt_kde)
        ])
        
        if len(samples) > N_rec:
            indice = torch.multinomial(self.ones(len(samples)), N_rec)
            samples = samples[indice]
        return samples
