import torch
import warnings
from torch.distributions.multivariate_normal import MultivariateNormal


class Utils:
    def __init__(self, device):
        """
        Args:
           - device: torch.device, cpu or cuda
        """
        self.eps = -torch.sqrt(torch.tensor(torch.finfo().max)).item()
        self.gpu_lim = int(5e5)
        self.max_iter = 10
        self.device = device

    def remove_anomalies(self, y):
        """
        Args:
           - y: torch.tensor, observations

        Returns:
           - y: torch.tensor, observations whose anomalies have been removed.
        """
        y[y.isnan()] = self.eps
        y[y.isinf()] = self.eps
        y[y < self.eps] = self.eps
        return y

    def remove_anomalies_uniform(self, X, uni_min, uni_max):
        """
        Args:
           - X: torch.tensor, inputs
           - uni_min: torch.tensor, the minimum limit values of uniform distribution
           - uni_max: torch.tensor, the maximum limit values of uniform distribution

        Returns:
           - idx: bool, indices where the inputs X do not exceed the min-max limits
        """
        logic = torch.sum(torch.stack([torch.logical_or(
            X[:, i] < uni_min[i],
            X[:, i] > uni_max[i],
        ) for i in range(X.size(1))]), axis=0)
        return (logic == 0)

    def is_psd(self, mat):
        """
        Args:
           - mat: torch.tensor, symmetric matrix

        Returns:
           - flag: bool, flag to judge whether or not the given matrix is positive semi-definite
        """
        return bool((mat == mat.T).all() and (torch.linalg.eig(mat)[0].real >= 0).all())
        
    def make_cov_psd(self, cov):
        """
        Args:
           - cov: torch.tensor, covariance matrix of multivariate normal distribution

        Returns:
           - cov: torch.tensor, covariance matrix of multivariate normal distribution
        """
        if self.is_psd(cov):
            return cov
        else:
            warnings.warn("Estimated covariance matrix was not positive semi-definite. Conveting...")
            cov = torch.nan_to_num(cov)
            cov = torch.sqrt(cov * cov.T)
            if not self.is_psd(cov):
                n_dim = cov.size(0)
                r_increment = 2
                jitter = torch.ones(n_dim).to(self.device) * 1e-5
                n_iter = 0
                while not self.is_psd(cov):
                    cov[range(n_dim), range(n_dim)] += jitter
                    jitter *= r_increment
                    n_iter += 1
                    if n_iter > self.max_iter:
                        cov = cov.diag().diag()
                        break
            return cov

    def safe_mvn_register(self, mu, cov):
        """
        Args:
           - mu: torch.tensor, mean vector of multivariate normal distribution
           - cov: torch.tensor, covariance matrix of multivariate normal distribution

        Returns:
           - mvn: torch.distributions, function of multivariate normal distribution
        """
        cov = self.make_cov_psd(cov)
        return MultivariateNormal(mu, cov)
        
    def safe_mvn_prob(self, mu, cov, X):
        """
        Args:
           - mu: torch.tensor, mean vector of multivariate normal distribution
           - cov: torch.tensor, covariance matrix of multivariate normal distribution
           - X: torch.tensor, the locations that we wish to calculate the probability density values

        Returns:
           - pdf: torch.tensor, the probability density values at given locations X.
        """
        mvn = self.safe_mvn_register(mu, cov)
        if X.size(0) > self.gpu_lim:
            warnings.warn("The matrix size exceeds the GPU limit. Splitting.")
            n_split = torch.tensor(X.size(0) / self.gpu_lim).ceil().long()
            _X = torch.tensor_split(X, n_split)
            Npdfs = torch.cat(
                [
                    mvn.log_prob(_X[i]).exp()
                    for i in range(n_split)
                ]
            )
        else:
            Npdfs = mvn.log_prob(X).exp()
        return Npdfs
