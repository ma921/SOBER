import torch
import warnings

class WeightsStabiliser:
    def __init__(
        self,
        eps=torch.finfo().eps,
        thresh=5,
    ):
        """
        A class of functions that stabilise the weight-related computations
        
        Args:
        - eps_weights: float, the machine epsilon (the smallest number of floating point).
                       Default: torch.finfo().eps
        - thresh: int, the number of non-zero weights which regrads anomalies.
        """
        self.eps_weights = eps
        self.thresh = thresh
        
    def cleansing_weights(self, weights):
        """
        Remove anomalies from the computed weights
        
        Args:
        - weights: torch.tensor, weights
        
        Return:
        - weights: torch.tensor, the cleaned weights
        """
        weights[weights < self.eps_weights] = 0
        weights[weights.isinf()] = self.eps_weights
        weights[weights.isnan()] = self.eps_weights
        if not weights.sum() == 0:
            weights /= weights.sum()
        else:
            weights = torch.ones_like(weights)/len(weights)
        return weights.detach()
    
    def check_weights(self, weights):
        """
        Check weights anomalies
        
        Args:
        - weights: torch.tensor, weights
        
        Return:
        - weights: torch.tensor, the cleaned weights
        """
        if weights.sum() == 0:
            return False
        elif len(weights.unique()) < self.thresh:
            return False
        else:
            return True
        
    def weighted_resampling(self, weights, n_nys):
        """
        Weighted resampling.
        len(weights) > n_nys should be satisfied.
        
        Args:
        - weights: torch.tensor, weights
        - n_nys: int, the number of resamples
        
        Return:
        - idx_nys: torch.tensor, the indices where the resamples locate.
        """
        n_positive_weights = (weights > 0).sum()
        if n_positive_weights > n_nys:
            idx_nys = torch.multinomial(weights, n_nys)
        else:
            idx_positive = torch.arange(len(weights))[weights > 0]
            idx_rand = torch.randperm(len(weights))[:int(n_nys - n_positive_weights)]
            idx_nys = torch.cat([idx_positive, idx_rand])
            warnings.warn("Non-zero weights are fewer than n_Nys: "+str(idx_nys.sum()))
        return idx_nys
    
    def deweighted_resampling(self, weights, n_samples):
        """
        Uniform resampling from weighted samples
        
        Args:
        - weights: torch.tensor, the unnormalised weights
        - n_samples, int, the number of uniform samples
        
        Return:
        - indice: torch.tensor, the indices of the selected uniform samples
        """
        weights_inv = (1 / weights)
        weights_inv = self.cleansing_weights(weights_inv)
        indice = self.weighted_resampling(weights_inv, n_samples)
        return indice
    
    def kmeans_resampling(self, X, n_clusters=100):
        _, X_sparse = KMeans(X, n_clusters)
        return X_sparse


def KMeans(x, K=10, Niter=10):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    N, D = x.shape  # Number of samples, dimension of the ambient space
    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    return cl, c