import copy
import torch
import torch.distributions as D
from torch.quasirandom import SobolEngine
#from rdkit.Chem import MolFromSmiles, AllChem
import numpy as np
import pandas as pd

class Uniform:
    def __init__(self, mins, maxs, n_dims):
        """
        Uniform prior class
        
        Args:
        - mins: torch.tensor, the lower bounds of continuous variables
        - maxs: torch.tensor, the upper bounds of continuous variables
        - n_dims: int, the number of dimensions
        """
        self.mins = mins
        self.maxs = maxs
        self.n_dims = n_dims
        self.type = "uniform"
        
    def sample(self, n_samples, qmc=True):
        """
        Sampling from Uniform prior
        
        Args:
        - n_samples: int, the number of initial samples
        - qmc: bool, sampling from Sobol sequence if True, otherwise simply Monte Carlo sampling.
        
        Return:
        - samples: torch.tensor, the samples from uniform prior
        """
        if qmc:
            random_samples = SobolEngine(self.n_dims, scramble=True).draw(n_samples)
        else:
            random_samples = torch.rand(n_samples, self.n_dims)
       
        return self.mins.unsqueeze(0) + (self.maxs - self.mins).unsqueeze(0) * random_samples
    
    def pdf(self, samples):
        """
        The probability density function (PDF) over samples
        
        Args:
        - samples: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over samples
        """
        _pdf = torch.ones(samples.size(0)) * (1/(self.maxs - self.mins)).prod()
        _ood = torch.logical_or(
            (samples >= self.maxs).any(axis=1), 
            (samples <= self.mins).any(axis=1),
        ).logical_not()
        return _pdf * _ood
    
    def logpdf(self, samples):
        """
        The log probability density function (PDF) over samples
        
        Args:
        - samples: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the log PDF over samples
        """
        _logpdf = torch.ones(samples.size(0)) * (1/(self.maxs - self.mins)).prod().log()
        _ood = torch.logical_or(
            (samples >= self.maxs).any(axis=1), 
            (samples <= self.mins).any(axis=1),
        ).logical_not()
        return _logpdf * _ood
    
class Gaussian:
    def __init__(self, mu, cov):
        """
        Gaussian prior class
        
        Args:
        - mu: torch.tensor, the mean vector of Gaussian distribution
        - cov: torch.tensor, the covariance matrix of Gaussian distribution
        """
        self.mu = mu
        self.cov = cov
        self.mvn = D.MultivariateNormal(self.mu, self.cov)
        self.type = "gaussian"
        
    def sample(self, n_samples):
        """
        Sampling from Gaussian prior
        
        Args:
        - n_samples: int, the number of initial samples
        
        Return:
        - samples: torch.tensor, the samples from Gaussian prior
        """
        return self.mvn.sample(torch.Size([n_samples]))
    
    def pdf(self, x):
        """
        The probability density function (PDF) over x
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over x
        """
        return self.mvn.log_prob(x).exp()

class CategoricalPrior:
    def __init__(self, n_dims, _min, _max, n_discrete):
        """
        Categorical prior class
        
        Args:
        - n_dims: int, the number of dimensions
        - _min: int, the lower bound of categorical variables
        - _max: int, the upper bound of categorical variables
        - n_discrete: int, the number of categories for each dimension
        """
        self.n_dims = n_dims
        self.min = _min
        self.max = _max
        self.n_discrete = n_discrete
        self.type = "categorical"
        self.set_prior()
        
    def set_prior(self):
        """
        Set parameters and functions
        """
        weights = torch.ones(self.n_discrete) / self.n_discrete
        self.pmf = weights.unique()[0]
        self.cat = D.Categorical(weights.repeat(self.n_dims, 1))
        self.discrete_candidates = torch.linspace(self.min, self.max, self.n_discrete)
    
    def sample(self, n_samples):
        """
        Sampling from categorical prior
        
        Args:
        - n_samples: int, the number of samples
        
        Return:
        - samples: torch.tensor, random samples from categorical distribution
        """
        indices = self.cat.sample(torch.Size([n_samples]))
        return self.discrete_candidates[indices]
    
    def sample_both(self, n_samples):
        """
        Sampling both categorical values and indices from categorical prior
        
        Args:
        - n_samples: int, the number of samples
        
        Return:
        - samples: torch.tensor, random samples from categorical distribution
        - indices: torch.tensor, indices of random samples
        """
        indices = self.cat.sample(torch.Size([n_samples]))
        return self.discrete_candidates[indices], indices
    
    def pdf(self, x):
        """
        The probability mass function (PMF) over x
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pmfs: torch.tensor, the PMF over x
        """
        return (torch.ones(x.size()) * self.pmf).prod(axis=1)
    
class BinaryPrior:
    def __init__(self, n_dims):
        """
        Bernoulli (Binary) prior class
        
        Args:
        - n_dims: int, the number of dimensions
        """
        self.n_dims = n_dims
        self.type = "bernoulli"
        self.prior_binary = D.Bernoulli(torch.ones(self.n_dims) * 0.5)
        
    def sample(self, n_samples):
        """
        Sampling from Bernoulli prior
        
        Args:
        - n_samples: int, the number of samples
        
        Return:
        - samples: torch.tensor, random samples from Bernoulli distribution
        """
        return self.prior_binary.sample(torch.Size([n_samples]))
    
    def pdf(self, samples):
        """
        The probability mass function (PMF) over samples
        
        Args:
        - samples: torch.tensor, the input where to compute PDF
        
        Return:
        - pmfs: torch.tensor, the PMF over samples
        """
        return self.prior_binary.log_prob(samples).exp().prod(axis=1)
    
    def logpdf(self, samples):
        """
        The log probability mass function (PMF) over samples
        
        Args:
        - samples: torch.tensor, the input where to compute PDF
        
        Return:
        - pmfs: torch.tensor, the log PMF over samples
        """
        return self.prior_binary.log_prob(samples).sum(axis=1)

class MixedBinaryPrior:
    def __init__(self, n_dims_cont, n_dims_binary, _min, _max):
        """
        Mixed prior of Bernoulli and uniform distributions
        
        Args:
        - n_dims_cont: int, the number of dimensions for continuous variables
        - n_dims_binary: int, the number of dimensions for binary variables
        - _min: int, the lower bound of continuous variables
        - _max: int, the upper bound of continuous variables
        """
        self.n_dims_cont = n_dims_cont
        self.n_dims_binary = n_dims_binary
        self.min = _min
        self.max = _max
        self.type = "mixedbinary"
        self.set_prior()
        
    def set_prior(self):
        """
        Set mixed prior
        """
        mins = self.min * torch.ones(self.n_dims_cont)
        maxs = self.max * torch.ones(self.n_dims_cont)
        self.prior_cont = Uniform(mins, maxs, self.n_dims_cont)
        self.prior_binary = BinaryPrior(self.n_dims_binary)
        
    def sample(self, n_samples):
        """
        Sampling from mixed prior
        
        Args:
        - n_samples: int, the number of samples
        
        Return:
        - samples: torch.tensor, random samples from mixed distribution
        """
        samples_cont = self.prior_cont.sample(n_samples)
        samples_binary = self.prior_binary.sample(n_samples)
        return torch.hstack([samples_cont, samples_binary])
    
    def pdf(self, x):
        """
        The probability density function (PDF) over x
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over samples
        """
        x_cont = x[:, :self.n_dims_cont]
        x_binary = x[:, self.n_dims_cont:]
        pdf_cont = self.prior_cont.pdf(x_cont)
        pdf_binary = self.prior_binary.pdf(x_binary)
        return pdf_cont * pdf_binary
    
    def logpdf(self, x):
        """
        The log probability density function (PDF) over x
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the log PDF over samples
        """
        x_cont = x[:, :self.n_dims_cont]
        x_binary = x[:, self.n_dims_cont:]
        pdf_cont = self.prior_cont.logpdf(x_cont)
        pdf_binary = self.prior_binary.logpdf(x_binary)
        return pdf_cont + pdf_binary

class MixedPrior:
    def __init__(self, n_dims_cont, n_dims_disc, n_discrete, _min, _max):
        """
        Mixed prior of categorical and uniform distributions
        
        Args:
        - n_dims_cont: int, the number of dimensions for continuous variables
        - n_dims_binary: int, the number of dimensions for categorical variables
        - n_discrete: int, the number of categories for each dimension
        - _min: int, the lower bound of continuous variables
        - _max: int, the upper bound of continuous variables
        """
        self.n_dims_cont = n_dims_cont
        self.n_dims_disc = n_dims_disc
        self.n_discrete = n_discrete
        self.min = _min
        self.max = _max
        self.type = "mixedcategorical"
        self.set_prior()
        
    def set_prior(self):
        """
        Set mixed prior
        """
        mins = self.min * torch.ones(self.n_dims_cont)
        maxs = self.max * torch.ones(self.n_dims_cont)
        self.prior_cont = Uniform(mins, maxs, self.n_dims_cont, n_samples=self.n_samples)
        self.prior_disc = CategoricalPrior(self.n_dims_disc, self.min, self.max, self.n_discrete, n_samples=self.n_samples)
        
    def sample(self, n_samples):
        """
        Sampling from mixed prior
        
        Args:
        - n_samples: int, the number of samples
        
        Return:
        - samples: torch.tensor, random samples from mixed distribution
        """
        samples_cont = self.prior_cont.sample(n_samples)
        samples_disc = self.prior_disc.sample(n_samples)
        return torch.hstack([samples_cont, samples_disc])
    
    def sample_both(self, n_samples):
        """
        Drawing both samples and indices from mixed prior
        
        Args:
        - n_samples: int, the number of samples
        
        Return:
        - samples: torch.tensor, random samples from categorical distribution
        - indices: torch.tensor, indices of categorical samples
        """
        samples_cont = self.prior_cont.sample(n_samples)
        samples_disc, indices = self.prior_disc.sample_both(n_samples)
        return torch.hstack([samples_cont, samples_disc]), torch.hstack([samples_cont, indices])
    
    def pdf(self, x):
        """
        The probability density function (PDF) over x
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over samples
        """
        x_cont = x[:, :self.n_dims_cont]
        x_disc = x[:, self.n_dims_cont:]
        pdf_cont = self.prior_cont.pdf(x_cont)
        pdf_disc = self.prior_disc.pdf(x_disc)
        return pdf_cont * pdf_disc