import torch
from ._gp import predict
from ._rchq import recombination


class KernelQuadrature:
    def __init__(
        self, 
        prior, 
        gp,
        n_rec=100000,
        n_nys=1000,
        n_quad=1000,
    ):
        """
        Args:
           - n_rec: int, subsampling size for kernel recombination
           - nys_ratio: float, subsubsampling ratio for Nystrom.
           - n_nys: int, number of Nystrom samples; int(nys_ratio * n_rec)
           - n_quad: int, number of kernel recombination subsamples; int(quad_ratio * n_rec)
           - batch_size: int, batch size
           - sampler: function of samples = function(n_samples)
           - kernel: function of covariance_matrix = function(X, Y). Positive semi-definite Gram matrix (a.k.a. kernel)
           - device: torch.device, cpu or cuda
           - mean_predict: function of mean = function(x), the function that returns the predictive mean at given x
        """
        self.n_rec = n_rec
        self.n_nys = n_nys
        self.n_quad = n_quad
        self.prior = prior
        self.gp = gp
        #self.kernel = self.gp.fspace_kernel
        self.kernel = gp.gspace_kernel
        self.device = device
        #self.mean_predict = self.gp.fspace_mean_predict
        self.mean_predict = self.gp.gspace_mean_predict

    def rchq(self, pts_nys, pts_rec, w_IS, batch_size, kernel):
        """
        Args:
            - pts_nys: torch.tensor, subsamples for low-rank approximation via Nyström method
            - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination
            - w_IS: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
            - batch_size: int, batch size
            - kernel: function of covariance_matrix = function(X, Y). Positive semi-definite Gram matrix (a.k.a. kernel)

        Returns:
            - x: torch.tensor, the sparcified samples from pts_rec. The number of samples are determined by self.batch_size
            - w: torch.tensor, the positive weights for kernel quadrature as discretised summation.
        """
        idx, w = recombination(
            pts_rec,
            pts_nys,
            batch_size,
            kernel,
            self.device,
            init_weights=w_IS,
        )
        x = pts_rec[idx]
        return x, w

    def quadrature(self):
        """
        Returns:
            - EZy: float, the mean of the evidence
            - VarZy: float, the variance of the evidence
        """
        pts_nys, pts_rec, w_IS = self.sampler(self.n_quad)
        X, w = self.rchq(pts_nys, pts_rec, w_IS, self.batch_size, self.kernel)
        EZy = (w @ self.mean_predict(X))
        VarZy = (w @ self.kernel(X, X) @ w)
        if EZy <= 0:
            self.logEZy = self.gp.beta.item()
        else:
            self.logEZy = (EZy.log() + self.gp.beta).item()
        logVarZy = VarZy.abs().log().item()
        #logVarZy = (VarZy.abs().log() + 2 * self.gp.beta).item()
        print("logE[Z|y]: " + str(self.logEZy) + "  logVar[Z|y]: " + str(logVarZy))
        return self.logEZy, logVarZy
