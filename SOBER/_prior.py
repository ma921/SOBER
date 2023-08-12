import copy
import torch
import torch.distributions as D
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from ._utils import TensorManager
from ._tmvn import TruncatedMVN
from .mvnorm import multivariate_normal_cdf as Phi


class BasePrior(ABC, TensorManager):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def sample(self, X):
        r"""Sampling from the prior"""
        pass
    
    @abstractmethod
    def pdf(self, X):
        r"""Return the probability density function of the prior"""
        pass

class Uniform(BasePrior):
    def __init__(self, bounds):
        """
        Uniform prior class
        
        Args:
        - bounds: torch.tensor, the lower and upper bounds for each dimension
        """
        super().__init__() # call TensorManager
        self.bounds = self.standardise_tensor(bounds)
        self.n_dims = self.bounds.shape[1]
        self.type = "continuous"
        
    def sample(self, n_samples, qmc=True):
        """
        Sampling from Uniform prior
        
        Args:
        - n_samples: int, the number of initial samples
        - qmc: bool, sampling from Sobol sequence if True, otherwise simply Monte Carlo sampling.
        
        Return:
        - samples: torch.tensor, the samples from uniform prior
        """
        random_samples = self.rand(self.n_dims, n_samples, qmc=qmc)
        samples = self.bounds[0].unsqueeze(0) + (
            self.bounds[1] - self.bounds[0]
        ).unsqueeze(0) * random_samples
        return samples
    
    def pdf(self, samples):
        """
        The probability density function (PDF) over samples
        
        Args:
        - samples: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over samples
        """
        _pdf = self.ones(len(samples)) * (1/(self.bounds[1] - self.bounds[0])).prod()
        _ood = torch.logical_or(
            (samples >= self.bounds[1]).any(axis=1), 
            (samples <= self.bounds[0]).any(axis=1),
        ).logical_not()
        return self.standardise_tensor(_pdf * _ood)
    
    def logpdf(self, samples):
        """
        The log probability density function (PDF) over samples
        
        Args:
        - samples: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the log PDF over samples
        """
        _logpdf = self.ones(len(samples)) * (1/(self.bounds[1] - self.bounds[0])).prod().log()
        _ood = torch.logical_or(
            (samples >= self.bounds[1]).any(axis=1), 
            (samples <= self.bounds[0]).any(axis=1),
        ).logical_not()
        return self.standardise_tensor(_logpdf * _ood)
    
class Gaussian(BasePrior):
    def __init__(self, mu, cov):
        """
        Gaussian prior class.
        
        Args:
        - mu: torch.tensor, the mean vector of Gaussian distribution
        - cov: torch.tensor, the covariance matrix of Gaussian distribution
        """
        super().__init__() # call TensorManager
        self.n_dims = len(mu)
        self.mvn = D.MultivariateNormal(
            self.standardise_tensor(mu), 
            self.standardise_tensor(cov),
        )
        self.type = "continuous"
    
    def sample(self, n_samples):
        """
        Sampling from Gaussian prior
        
        Args:
        - n_samples: int, the number of initial samples
        
        Return:
        - samples: torch.tensor, the samples from Gaussian prior
        """
        samples = self.mvn.sample(torch.Size([n_samples]))
        return samples
    
    def pdf(self, x):
        """
        The probability density function (PDF) over x.
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over x
        """
        pdfs = self.mvn.log_prob(x).exp()
        return pdfs
        
class TruncatedGaussian(BasePrior):
    def __init__(self, mu, cov, bounds):
        """
        Truncated Gaussian prior class.
        
        Args:
        - mu: torch.tensor, the mean vector of Gaussian distribution
        - cov: torch.tensor, the covariance matrix of Gaussian distribution
        """
        super().__init__() # call TensorManager
        self.n_dims = len(mu)
        self.mvn = D.MultivariateNormal(
            self.standardise_tensor(mu), 
            self.standardise_tensor(cov),
        )
        self.type = "continuous"
        self.bounds = self.standardise_tensor(bounds)
        p_lb = Phi(bounds[0], loc=mu, covariance_matrix=cov)
        p_ub = Phi(bounds[1], loc=mu, covariance_matrix=cov)
        self.constant = self.standardise_tensor(p_ub - p_lb)
        self.tmvn = TruncatedMVN(mu, cov, bounds)
    
    def sample(self, n_samples):
        """
        Sampling from Gaussian prior
        
        Args:
        - n_samples: int, the number of initial samples
        
        Return:
        - samples: torch.tensor, the samples from Gaussian prior
        """
        samples = self.tmvn.sample(n_samples)
        return self.standardise_tensor(samples)
    
    def pdf(self, x):
        """
        The probability density function (PDF) over x.
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over x
        """
        pdfs = self.mvn.log_prob(x).exp() / self.constant
        # Thresholding out-of-bound samples
        indices_min = (x < self.bounds[0]).any(axis=1)
        indices_max = (x > self.bounds[1]).any(axis=1)
        pdfs[indices_min] = self.zeros(len(x))[indices_min]
        pdfs[indices_max] = self.zeros(len(x))[indices_max]
        return pdfs

class CategoricalPrior(BasePrior):
    def __init__(self, categories):
        """
        Categorical prior class
        
        Args:
        - categories: list, the number of dimensions x number of categories
        """
        super().__init__() # call TensorManager
        self.type = "categorical"
        self.set_prior(categories)
        
    def set_prior(self, categories):
        """
        Set parameters and functions
        
        Args:
        - categories: list, the number of dimensions x number of categories
        """
        self.categories = [self.tensor(dim) for dim in categories]
        self.n_dims = len(categories)
        self.n_categories = self.tensor(
            [len(category) for category in self.categories]
        ).long()
        self.weights = [self.ones(n_category) * 0.5 for n_category in self.n_categories]
        self.initialise()
        
    def initialise(self):
        """
        Reset the weights of the categorical distribution classes
        """
        self.cats = [D.Categorical(weight) for weight in self.weights]
    
    def find_corresponding_categories(self, indices):
        """
        Find the corresponding categories from the indices
        
        Args:
        - indices: torch.tensor, the indices for each dimension and category
        
        Return:
        - samples: torch.tensor, categories correspinding to the given indices
        """
        indexing_tensor = self.arange(indices.size(1))
        samples = torch.stack([
            self.categories[dim][indices[:, dim]] for dim in indexing_tensor
        ], dim=1)
        return samples
    
    def sample_both(self, n_samples):
        """
        Sampling both categorical values and indices from categorical prior
        
        Args:
        - n_samples: int, the number of samples
        
        Return:
        - samples: torch.tensor, random samples from categorical distribution
        - indices: torch.tensor, indices of random samples
        """
        indices = torch.vstack([cat.sample(torch.Size([n_samples])) for cat in self.cats]).T
        samples = self.find_corresponding_categories(indices)
        return self.standardise_device(samples), indices
    
    def sample(self, n_samples):
        """
        Sampling from categorical prior
        
        Args:
        - n_samples: int, the number of samples
        
        Return:
        - samples: torch.tensor, random samples from categorical distribution
        """
        samples, _ = self.sample_both(n_samples)
        return samples
    
    def logpdf(self, x):
        """
        The log probability mass function (PMF) over x
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pmfs: torch.tensor, the PMF over x
        """
        return torch.vstack([
            self.cats[dim].log_prob(x[:,dim]) for dim in range(self.n_dims)
        ]).sum(axis=0)
    
    def pdf(self, x):
        """
        The probability mass function (PMF) over x
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pmfs: torch.tensor, the PMF over x
        """
        return self.logpdf(x).exp()
    
class BinaryPrior(BasePrior):
    def __init__(self, n_dims):
        """
        Bernoulli (Binary) prior class
        
        Args:
        - n_dims: int, the number of dimensions
        """
        super().__init__() # call TensorManager
        self.n_dims = n_dims
        self.type = "binary"
        self.prior_binary = D.Bernoulli(self.ones(self.n_dims) * 0.5)
        
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

class MixedBinaryPrior(BasePrior):
    def __init__(
        self, 
        n_dims_cont, 
        n_dims_binary,
        bounds,
        continous_first=True,
    ):
        """
        Mixed prior of Bernoulli and uniform distributions
        
        Args:
        - n_dims_cont: int, the number of dimensions for continuous variables
        - n_dims_binary: int, the number of dimensions for binary variables
        - bounds: torch.tensor, the lower and upper bounds for each dimension only for continuous variables
        - continous_first: bool, [cont, binary] if true, otherwise [binary, cont].
        """
        super().__init__() # call TensorManager
        self.n_dims_cont = n_dims_cont
        self.n_dims_binary = n_dims_binary
        self.bounds = self.standardise_tensor(bounds)
        self.continous_first = continous_first
        self.type = "mixedbinary"
        self.set_prior()
        
    def set_prior(self):
        """
        Set mixed prior
        """
        self.prior_cont = Uniform(self.bounds)
        self.prior_binary = BinaryPrior(self.n_dims_binary)
        
    def separate_samples(self, x):
        """
        Separate mixed variables to each type
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - x_cont: torch.tensor, continuous variables
        - x_binary: torch.tensor, binary variables
        """
        if self.continous_first:
            x_cont = x[:, :self.n_dims_cont]
            x_binary = x[:, self.n_dims_cont:]
        else:
            x_binary = x[:, :self.n_dims_binary]
            x_cont = x[:, self.n_dims_binary:]
        return x_cont, x_binary
        
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
        if self.continous_first:
            return torch.hstack([samples_cont, samples_binary])
        else:
            return torch.hstack([samples_binary, samples_cont])
    
    def pdf(self, x):
        """
        The probability density function (PDF) over x
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over samples
        """
        x_cont, x_binary = self.separate_samples(x)
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
        x_cont, x_binary = self.separate_samples(x)
        pdf_cont = self.prior_cont.logpdf(x_cont)
        pdf_binary = self.prior_binary.logpdf(x_binary)
        return pdf_cont + pdf_binary

class MixedCategoricalPrior(BasePrior):
    def __init__(self, n_dims_cont, n_dims_disc, categories, bounds, continous_first=True):
        """
        Mixed prior of categorical and uniform distributions
        
        Args:
        - n_dims_cont: int, the number of dimensions for continuous variables
        - n_dims_binary: int, the number of dimensions for categorical variables
        - n_discrete: int, the number of categories for each dimension
        - bounds: torch.tensor, the lower and upper bounds for each dimension only for continuous variables
        - categories: torch.tensor, the categories to select for categorical variables.
        - continous_first: bool, [cont, categorical] if true, otherwise [categorical, cont].
        """
        super().__init__() # call TensorManager
        self.n_dims_cont = n_dims_cont
        self.n_dims_disc = n_dims_disc
        self.categories = self.standardise_tensor(categories)
        self.bounds = self.standardise_tensor(bounds)
        self.continous_first = continous_first
        self.type = "mixedcategorical"
        self.set_prior()
        
    def set_prior(self):
        """
        Set mixed prior
        """
        self.prior_cont = Uniform(self.bounds)
        self.prior_disc = CategoricalPrior(self.categories)
        
    def separate_samples(self, x):
        """
        Separate mixed variables to each type
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - x_cont: torch.tensor, continuous variables
        - x_disc: torch.tensor, categorical variables
        """
        if self.continous_first:
            x_cont = x[:, :self.n_dims_cont]
            x_disc = x[:, self.n_dims_cont:]
        else:
            x_disc = x[:, :self.n_dims_disc]
            x_cont = x[:, self.n_dims_disc:]
        return x_cont, x_disc
        
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
        if self.continous_first:
            return torch.hstack([samples_cont, samples_disc])
        else:
            return torch.hstack([samples_disc, samples_cont])
    
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
        if self.continous_first:
            return (
                torch.hstack([samples_cont, samples_disc]), 
                torch.hstack([samples_cont, indices]),
            )
        else:
            return (
                torch.hstack([samples_disc, samples_cont]), 
                torch.hstack([indices, samples_cont]),
            )
    
    def pdf(self, x):
        """
        The probability density function (PDF) over x
        
        Args:
        - x: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over samples
        """
        x_cont, x_disc = self.separate_samples(x)
        pdf_cont = self.prior_cont.pdf(x_cont)
        pdf_disc = self.prior_disc.pdf(x_disc)
        return pdf_cont * pdf_disc

class DatasetPrior(BasePrior):
    def __init__(
        self,
        features,
        true_targets,
    ):
        """
        Dataset prior for which all list of possible candidates are given as dataset
        
        Args:
        - features: torch.tensor, the binary inputs
        - true_targets: torch.tensor, the objective to maximize
        """
        super().__init__() # call TensorManager
        self.available_index = self.arange(len(features))
        self.features = self.standardise_tensor(features)
        self.true_targets = self.standardise_tensor(true_targets)
        self.reset_indices(self.available_index)
        self.type = "dataset"
        
    def reset_indices(self, available_index):
        """
        Reset the indices of dataset
        
        Args:
        - available_index: torch.tensor, the available indices that the queried indices are removed.
        """
        self.n_available = available_index.shape[0]
        self.features = self.features[available_index]
        self.true_targets = self.true_targets[available_index]
        self.available_index = self.arange(self.n_available)
        
    def set_substract(self, A, B):
        """
        Substracting the set B from the set A, where len(A) > len(B)
        
        Args:
        - A: torch.tensor, the set of indices.
        - B: torch.tensor, the set of indices.
        
        Return:
        - A-B: torch.tensor, the substracted set (A/B)
        """
        mask = self.ones(A.shape).to(torch.bool)
        mask[B] = 0
        return torch.masked_select(A, mask)
        
    def remove_sampled_index(self, idx_sampled):
        """
        Remove the sampled indices
        
        Args:
        - idx_sampled: torch.tensor, the sampled indices from the available indices.
        """
        available_index = self.set_substract(self.available_index, idx_sampled)
        self.reset_indices(available_index)
    
    def query(self, idx_cand):
        """
        Query Y at given indices.
        Then, update the internal dataset to delete the drawn samples.
        
        Args:
        - X_cand: torch.tensor, the features to query.
        
        Return:
        - Y: torch.tensor, the true values
        """
        Y = self.true_targets[idx_cand]
        self.remove_sampled_index(idx_cand)
        return Y
    
    def sample(self, n_sample):
        """
        Sample both X and Y with the size of n_sample.
        Then, update the internal dataset to delete the drawn samples. 
        
        Args:
        - n_sample: int, the number of samples.
        
        Return:
        - X: torch.tensor, the features.
        - Y: torch.tensor, the true values.
        """
        idx_sampled = self.randperm(self.n_available)[:n_sample]
        X = self.features[idx_sampled]
        Y = self.true_targets[idx_sampled]
        self.remove_sampled_index(idx_sampled)
        return X, Y
    
    def sample_feature(self, n_sample):
        """
        Sample X with the size of n_sample
        
        Args:
        - n_sample: int, the number of samples.
        
        Return:
        - X: torch.tensor, the features.
        """
        idx_sampled = self.randperm(self.n_available)[:n_sample]
        X = self.features[idx_sampled]
        return idx_sampled, X
    
    def available_candidates(self):
        """
        Sample all available X
        
        Return:
        - X: torch.tensor, the features.
        """
        return self.features
    
    def pdf(self, X):
        return self.ones(len(X)) / len(X)
