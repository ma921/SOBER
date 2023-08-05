from ._gp import predict
import torch.distributions as D


class PI:
    def __init__(self, model, label="lfi"):
        """
        Definition of pi (feasible resion).
        Select TS (Thompson Sampling) or LFI (likelihood-free inference)
        
        Args:
        - model: gpytorch.models, function of GP model
        - label: string, "ts" or "lfi"
        """
        self.model = model
        self.Xobs = model.train_inputs[0]    # training data X_obs
        self.eta = self.model.likelihood(self.model(self.Xobs)).loc.max().item() # current maximum
        self.label = label
    
    def lfi(self, X_cand, log=False):
        """
        Compute LFI at given locations X_cand
        
        Args:
        - X_cand: torch.tensor, inputs X where to compute LFI
        - log: bool: return log value if true, otherwise not.
        
        Return:
        - lfi: torch.tensor, the LFI values at X_cand
        """
        mu_pred, var_pred = predict(X_cand, self.model)
        lfi = D.Normal(0,1).cdf(
            (mu_pred - self.eta) / var_pred.sqrt()
        )
        if log:
            return (lfi + torch.finfo().eps).log()
        else:
            return lfi
    
    def __call__(self, X_cand, log=False):
        """
        Compute pi at given locations X_cand
        
        Args:
        - X_cand: torch.tensor, inputs X where to compute LFI
        - log: bool: return log value if true, otherwise not.
        
        Return:
        - pi: torch.tensor, the pi values at X_cand
        """
        if self.label == "ts":
            raise NotImplementedError("Not implemented yet")
        elif self.label == "lfi":
            return self.lfi(X_cand, log=log)
        else:
            raise ValueError("Label should be either 'ts' or 'lfi'.")
            
class PI_FBGP:
    def __init__(self, model):
        """
        Definition of pi (feasible resion) for fully Bayesian GP.
        
        Args:
        - model: gpytorch.models, function of GP model
        - label: string, "ts" or "lfi"
        """
        self.model = model
    
    def lfi(self, X_cand, log=False):
        """
        Compute LFI at given locations X_cand
        
        Args:
        - X_cand: torch.tensor, inputs X where to compute LFI
        - log: bool: return log value if true, otherwise not.
        
        Return:
        - lfi: torch.tensor, the LFI values at X_cand
        """
        mu_batch, var_batch = self.model.batch_predict(X_cand)
        lfi = D.Normal(0,1).cdf(
            (mu_batch - self.model.Theta_qd[:,0].unsqueeze(1))/ var_batch.sqrt()
        )
        lfi = self.model.w_qd @ lfi
        
        if log:
            return (lfi + torch.finfo().eps).log()
        else:
            return lfi
    
    def __call__(self, X_cand, log=False):
        """
        Compute pi at given locations X_cand
        
        Args:
        - X_cand: torch.tensor, inputs X where to compute LFI
        - log: bool: return log value if true, otherwise not.
        
        Return:
        - pi: torch.tensor, the pi values at X_cand
        """
        if self.label == "ts":
            raise NotImplementedError("Not implemented yet")
        elif self.label == "lfi":
            return self.lfi(X_cand, log=log)
        else:
            raise ValueError("Label should be either 'ts' or 'lfi'.")

class PI_BQ:
    def __init__(self, model):
        """
        Definition of pi (feasible resion) for Bayesian Quadrature GP model.
        For BQ modelling, see https://arxiv.org/abs/2210.17299
        
        Args:
        - model: gpytorch.models, function of GP model
        - label: string, "ts" or "lfi"
        """
        self.model = model
    
    def lfi(self, X_cand, log=False):
        """
        Compute LFI at given locations X_cand
        
        Args:
        - X_cand: torch.tensor, inputs X where to compute LFI
        - log: bool: return log value if true, otherwise not.
        
        Return:
        - lfi: torch.tensor, the LFI values at X_cand
        """
        mu_pred, var_pred = self.model.gspace_predict(X_cand)
        lfi = D.Normal(0,1).cdf(
            (mu_pred - 1) / var_pred.sqrt()
        )
        if log:
            return (lfi + torch.finfo().eps).log()
        else:
            return lfi
    
    def __call__(self, X_cand, log=False):
        """
        Compute pi at given locations X_cand
        
        Args:
        - X_cand: torch.tensor, inputs X where to compute LFI
        - log: bool: return log value if true, otherwise not.
        
        Return:
        - pi: torch.tensor, the pi values at X_cand
        """
        if self.label == "ts":
            raise NotImplementedError("Not implemented yet")
        elif self.label == "lfi":
            return self.lfi(X_cand, log=log)
        else:
            raise ValueError("Label should be either 'ts' or 'lfi'.")
    