from ._gp import predictive_covariance, predict_mean
import torch.distributions as D


class Kernel:
    def __init__(self, model, weighted=False):
        """
        Definition of kernel for recombination.
        
        Args:
        - model: gpytorch.models, function of GP model
        - weighted: bool, weighted kernel if True, otherwise normal predictive covariance
        """
        self.model = model
        self.weighted = weighted
    
    def __call__(self, x, y):
        """
        Compute the Gram matrix
        
        Return:
        - CLy: torch.tensor, the Gram matrix with the posterior predictive covariance
        """
        if self.weighted:
            return self.weighted_covariance(x, y)
        else:
            return predictive_covariance(x, y, self.model)
    
    def weighted_covariance(self, x, y):
        """
        Compute the mean weighted Gram matrix
        
        Return:
        - CLy: torch.tensor, the Gram matrix with the posterior predictive covariance
        """
        mu_x = predict_mean(x, self.model)
        mu_y = predict_mean(y, self.model)
        cov_xy = predictive_covariance(x, y, self.model)
        CLy = mu_x.unsqueeze(1) * cov_xy * mu_y.unsqueeze(0)
        return CLy
