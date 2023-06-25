import copy
import torch
import warnings
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from ._utils import Utils
from ._weights import WeightsStabiliser


class WeightedKernelDensityEstimation(WeightsStabiliser):
    def __init__(
        self,
        X,
        W,
        n_dims,
        n_kde=4096,
        bw_method='scott',
    ):
        """
        Class of weighted kernel density estimation with Gaussian kernel.
        
        Args:
        - X: torch.tensor, the observed data X
        - W: torch.tensor, the weights over X
        - n_dims: int, the number of dimensions
        - n_kde: int, the number of Gaussians for KDE.
        - bw_method: string, 'scott' or 'silverman'        
        """
        super().__init__(eps=0, thresh=n_kde) # WeightsStabiliser class initialisation
        self.n_dims = n_dims
        self.n_kde = min([n_kde, len(X)])
        self.bw_method = bw_method
        self.type = "continuous"

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.utils = Utils(device)
        self.initialisation(X, W)
    
    def initialisation(self, X, Y):
        """
        Estimate hyperparameters and sparse Gaussians
        
        Args:
        - X: torch.tensor, the observed data X
        - Y: torch.tensor, the unnormalised weights
        """
        if self.check_weights(Y):
            idx_accept = self.deweighted_sampling(Y, self.n_kde)
        else:
            idx_accept = torch.arange(Y.size(0))[self.cleansing_weights(Y) > 0]
            self.n_kde = len(idx_accept)
            if self.n_kde < 1:
                raise ValueError("Invalid weights")
        
        self.Xobs = X[idx_accept]
        self.weights = self.cleansing_weights(Y[idx_accept])
        self.n_kde = self.Xobs.size(0)
        self.set_bandwidth()
        self._compute_covariance()

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
        self.covariance = self.utils.make_cov_psd(self._data_covariance * self.bw.pow(2))

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

        Npdfs_AA = self.utils.safe_mvn_prob(
            torch.zeros(self.n_dims),
            self.covariance,
            x_AA,
        ).reshape(n_X, self.n_kde)
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
    
    def sample_from_Gaussian(self, mean, cov, cnt):
        """
        Sampling from each Gaussian
        
        Args:
        - mean: torch.tensor, the mean vector of Gaussian
        - cov: torch.tensor, the covariance matrix of Gaussian
        - cnt: int, the number of samples
        
        Return
        - samples: torh.tensor, samples from Gaussian
        """
        if cnt == 0:
            warnings.warn("invalid Gaussian in the kernel density estimation")
            return torch.tensor([]) #torch.zeros(0, len(mean))
        elif (cov == 0).all():
            warnings.warn("invalid Gaussian in the kernel density estimation")
            return torch.tensor([]) #torch.zeros(0, len(mean))
        else:
            cov = self.utils.make_cov_psd(cov)
            return MultivariateNormal(
                mean,
                cov,
            ).sample(torch.Size([cnt]))
    
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
            indice = torch.multinomial(torch.ones(len(samples)), N_rec)
            samples = samples[indice]
        elif len(samples) < N_rec:
            breakpoint()
        return samples
    