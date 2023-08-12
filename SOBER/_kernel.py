from ._gp import predictive_covariance, predict_mean
import torch.distributions as D

class Kernel:
    def __init__(self, model, mode="predictive_covariance"):
        """
        Definition of kernel for recombination.
        
        Args:
        - model: gpytorch.models, function of GP model
        - mode: string, select from ["predictive_covariance", "weighted_predictive_covariance", "kernel"]
        """
        self.model = model
        self.mode = mode
    
    def __call__(self, x, y):
        """
        Compute the Gram matrix of posterior predictive covariance
        
        Return:
        - CLy: torch.tensor, the Gram matrix with the posterior predictive covariance
        """
        if self.mode == "predictive_covariance":
            return predictive_covariance(x, y, self.model)
        elif self.mode == "weighted_predictive_covariance":
            return self.weighted_covariance(x, y)
        elif self.mode == "kernel":
            return self.model.covar_module.forward(x, y)
        else:
            raise ValueError('mode should be from ["predictive_covariance", "weighted_predictive_covariance", "kernel"]')
            
    
    def weighted_covariance(self, x, y):
        """
        Compute the mean weighted Gram matrix
        
        Return:
        - CLy: torch.tensor, the Gram matrix with the posterior predictive covariance
        """
        mu_x = predict_mean(x, self.model)
        mu_y = predict_mean(y, self.model)
        cov_xy = predictive_covariance(x, y, self.model)
        if len(mu_x.shape) == 1 and len(mu_y.shape) == 1:
            CLy = mu_x.unsqueeze(1) * cov_xy * mu_y.unsqueeze(0)
        else:
            CLy = mu_x.unsqueeze(1) * cov_xy * mu_y.unsqueeze(1)
        return CLy
