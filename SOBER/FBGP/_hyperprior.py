import copy
import torch
import torch.distributions as D


class RBFHyperPrior:
    def __init__(self, theta_map=None):
        """
        Hyperprior for the RBF kernel hyperparameters of FITBO Gaussian process model.
        Hyperprior is set as log-normal distribution. We set hyperprior on log-transformed space.
        While theta is a set of log-space hyperparmeters, Theta is a set of normal space hyperparamters
        theta = [eta, lik_var, lengthscale, outputscale]
        eta: the derivation from the current maximum of the observation, eta = y - max(Yobs)
        lik_var: the noise variance for the likelihood
        lengthscale: the lengthscale hyperparameter of RBF kernel
        outputscale: the outputscale hyperparameter of RBF kernel
        
        Args:
        - theta_map: torch.tensor or None, type-II MLE estimated hyperparameters.
        """
        self.initialise(theta_map)
        
    def initial_hyperprior(self, theta_map=None):
        """
        Initial hyperparameters recommended by the following paper.
        https://proceedings.neurips.cc/paper/2016/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html
        
        Args:
        - theta_map: torch.tensor or None, type-II MLE estimated hyperparameters.
        
        Return:
        - hypermu: the mean of log-space multivariate normal distribution
        - hypercov: the covariance matrix of log-space multivariate normal distribution
        """
        if theta_map == None:
            hypermu = torch.tensor([-2, 0.1, 0.1, 0.4])
            hyperstd = torch.tensor([0.7, 1, 0.7, 0.7])
        else:
            hypermu = torch.cat([
                torch.tensor([-2]),
                theta_map.log(),
            ])
            hyperstd = torch.tensor([0.1, 0.1, 0.1, 0.1])
        
        hypercov = hyperstd.pow(2).diag()
        return hypermu, hypercov
    
    def initialise(self, theta_map):
        """
        Initialise hyper-hyperparameters of hyperprior
        
        Args:
        - theta_map: torch.tensor or None, type-II MLE estimated hyperparameters.
        """
        self.hypermu, self.hypercov = self.initial_hyperprior(theta_map)
        self.mvn = D.MultivariateNormal(
            self.hypermu, self.hypercov,
        )
    
    def sample(self, n_samples):
        """
        Sampling from hyperprior
        
        Args:
        - n_samples: int
        
        Return:
        - hypersamples: torch.tensor, the hypersamples drawn from hyperprior
        """
        return self.mvn.sample(torch.Size([n_samples]))
    
    def pdf(self, hypersamples):
        """
        Compute the probability density function (PDF) values
        
        Args:
        - hypersamples: torch.tensor, the hypersamples of log-transformed hyperparameters
        
        Return:
        - pdfs: torch.tensor, the PDF values
        """
        return self.mvn.log_prob(hypersamples).exp()
