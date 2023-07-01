import torch
from torch.distributions.normal import Normal


class FBGPAcquisitionFunction:
    def __init__(self, model, label="MES"):
        """
        Acquisition functions defined by fully Bayesian Gaussian process (FBGP) model
        
        Args:
        - model: FullyBayesianGP class, a function of FBGP model.
        - label: string, select from ["EI", "UCB", "MES", "BQBC", "QBMGP"],
                 EI: expected improvement
                 UCB: upper confidence bound
                 MES: max-value entropy search
                 BQBC: Bayesian query-by-committee
                 QBMGP: Query by a mixture of Gaussian processes
        """
        self.model = model
        self.label = label
    
    def EI(self, mu_batch, var_batch):
        """
        Expected improvement acquisition function
        
        Args:
        - mu_batch: torch.tensor, a batch of predictive mean from FBGP model
        - var_batch: torch.tensor, a batch of predictive variance from FBGP model
        
        Return:
        - af: torch.tensor, the result of acquisition function computation
        """
        integrand = (mu_batch - self.model.Theta_qd[:,0].unsqueeze(1)) / var_batch.sqrt()
        Phi = Normal(0,1).cdf(integrand)
        phi = Normal(0,1).log_prob(integrand).exp()
        first = (mu_batch - self.model.Theta_qd[:,0].unsqueeze(1)) * Phi
        second = var_batch.sqrt() * phi
        return self.model.w_qd @ (first + second)

    def UCB(self, mu_batch, var_batch):
        """
        Upper confidence bound acquisition function
        
        Args:
        - mu_batch: torch.tensor, a batch of predictive mean from FBGP model
        - var_batch: torch.tensor, a batch of predictive variance from FBGP model
        
        Return:
        - af: torch.tensor, the result of acquisition function computation
        """
        Ey = self.model.w_qd @ mu_batch
        Vy = self.model.w_qd @ (var_batch + mu_batch.pow(2)) - Ey**2
        return Ey + Vy.sqrt()

    def FITBO(self, mu_batch, var_batch):
        """
        Max-value entropy search acquisition function.
        Approximated by FITBO formulation (https://arxiv.org/abs/1711.00673)
        
        Args:
        - mu_batch: torch.tensor, a batch of predictive mean from FBGP model
        - var_batch: torch.tensor, a batch of predictive variance from FBGP model
        
        Return:
        - af: torch.tensor, the result of acquisition function computation
        """
        Ey = self.model.w_qd @ mu_batch
        Vary = self.model.w_qd @ (var_batch + mu_batch.pow(2)) - Ey**2
        H1 = 0.5 * (2 * torch.pi * torch.e * (Vary + self.model.w_qd @ self.model.Theta_qd[:,1])).log()
        H2 = 0.5 * self.model.w_qd @ (2 * torch.pi * torch.e * (var_batch + self.model.Theta_qd[:,1].unsqueeze(1))).log()
        return H1 - H2

    def BQBC(self, mu_batch):
        """
        Bayesian query-by-committee acquisition function.
        (https://arxiv.org/abs/2205.10186)
        
        Args:
        - mu_batch: torch.tensor, a batch of predictive mean from FBGP model
        - var_batch: torch.tensor, a batch of predictive variance from FBGP model
        
        Return:
        - af: torch.tensor, the result of acquisition function computation
        """
        Ey = self.model.w_qd @ mu_batch
        return self.model.w_qd @ (mu_batch - Ey)

    def QBMGP(self, mu_batch, var_batch):
        """
        Query by a mixture of Gaussian processes acquisition function.
        (https://arxiv.org/abs/2205.10186)
        
        Args:
        - mu_batch: torch.tensor, a batch of predictive mean from FBGP model
        - var_batch: torch.tensor, a batch of predictive variance from FBGP model
        
        Return:
        - af: torch.tensor, the result of acquisition function computation
        """
        Ey = self.model.w_qd @ mu_batch
        Vy = self.model.w_qd @ (var_batch + mu_batch.pow(2)) - Ey**2
        return Vy + self.BQBC(mu_batch)
    
    def __call__(self, x):
        mu_batch, var_batch = self.model.batch_predict(x)
        if self.label == "EI":
            return self.EI(mu_batch, var_batch)
        elif self.label == "UCB":
            return self.UCB(mu_batch, var_batch)
        elif self.label == "MES":
            return self.FITBO(mu_batch, var_batch)
        elif self.label == "BQBC":
            return self.BQBC(mu_batch)
        elif self.label == "QBMGP":
            return self.QBMGP(mu_batch, var_batch)
        else:
            raise ValueError("Acquisition function type should be from ['EI', 'UCB','MES', 'BQBC', 'QBMGP']")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
