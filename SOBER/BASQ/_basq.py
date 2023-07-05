import torch
from .._rchq import recombination
from .._sampler import MixtureSampler
from .._utils import TensorManager

class BASQ(TensorManager):
    def __init__(
        self, 
        prior, 
        model,
        sober,
        ratio_wkde=1
    ):
        """
        inference of evidence and posterior by BASQ.
        See details in https://arxiv.org/abs/2206.04734
        
        Args:
           - prior: Prior class, prior distribution
           - model: ScaleMmltGP class, Bayesian quadrature model
           - sober: Sober class, SOBER model
           - ratio_wkde: float, the proportion to sample from pi
        """
        super().__init__()
        self.prior = prior
        self.update_model(model, sober, ratio_wkde=ratio_wkde)

    def update_model(self, model, sober, ratio_wkde=1):
        """
        Update model
        
        Args:
           - model: ScaleMmltGP class, Bayesian quadrature model
           - sober: Sober class, SOBER model
           - ratio_wkde: float, the proportion to sample from pi
        """
        self.kernel = model.gspace_kernel
        self.pred_mean = model.gspace_mean_predict
        self.beta = model.beta
        self.sampler = MixtureSampler(self.prior, sober, ratio_wkde=ratio_wkde)
        
    def quadrature(self, n_quad, n_nys_quad, n_res_quad):
        """
        Kernel quadrature for the estimation of marginal likelihood
        
        Args:
           - n_quad: int, sampling size for kernel recombination
           - n_nys_quad: int, number of samples for Nyström approximation
           - n_res_quad: int, number of kernel recombination subsamples
           
        Returns:
            - ELML: float, Expected log marginal likelihood
            - AVLML: float, Variance log marginal likelihood
        """
        X_cand = self.prior.sample(n_quad)
        w_IS = self.ones(n_quad) / n_quad
        X_nys = X_cand[:n_nys_quad]
        
        idx, w = recombination(
            X_cand,
            X_nys,
            n_res_quad,
            self.kernel,
            self.device,
            self.dtype,
            init_weights=w_IS,
        )
        x = X_cand[idx]
        
        # expected log marginal likelihood
        self.EML = w @ self.pred_mean(x)
        if self.EML <= 0:
            ELML = self.beta
            self.EML = self.beta.exp()
        else:
            ELML = self.EML.log() + self.beta
        # approximated variance of log marginal likelihod
        AVLML = (w @ self.kernel(x, x) @ w).abs().log() # + 2 * model.beta
        print(f"Expected log marginal likelihood: {ELML.item():.5e}")
        print(f"Variance log marginal likelihood: {AVLML.item():.5e}")
        return ELML.item(), AVLML.item()
    
    def posterior(self, x):
        """
        Probability density function of the estimated posterior
        
        Args:
           - x: torch.tensor, the input
           
        Returns:
            - posterior_pred: torch.tensor, the expected PDF of the estimated posterior
        """
        if hasattr(self, "EML"):
            likelihood_pred = self.pred_mean(x)
            likelihood_pred[likelihood_pred < 0] = 0
            if self.EML <= 0:
                raise ValueError("Evidence is not positive.")
            else:
                posterior_pred = likelihood_pred * self.prior.pdf(x) / self.EML
            return posterior_pred
        else:
            raise ValueError("Evidence has not yet computed.")
    
    def sampling_posterior(self, n_samples, ratio_super=100):
        """
        Approximately sampling from posterior via sequential importance resampling (SIR)
        
        Args:
           - n_samples: int, number of samples to draw
           - ratio_super: float, the ratio to supersample
           
        Returns:
            - samples: torch.tensor, the samples from the estimated posterior
        """
        n_supersamples = int(ratio_super * n_samples)
        samples = self.sampler.sample(n_supersamples)
        pdf_sampler = self.sampler.pdf(samples)
        pdf_posterior = self.posterior(samples)
        weights = pdf_posterior / pdf_sampler
        weights = self.sampler.sober.cleansing_weights(weights)
        idx = self.sampler.sober.weighted_resampling(weights.detach(), n_samples)
        samples = samples[idx]
        return samples
    
    def MAP(self, n_samples):
        """
        Maximum a posteriori (MAP) estimation
        
        Args:
           - n_samples: int, number of samples to draw
           
        Returns:
            - MAP: torch.tensor, the MAP value
        """
        samples = self.sampler.sample(n_samples)
        pdf_posterior = self.posterior(samples)
        MAP = samples[pdf_posterior.argmax()]
        return MAP
