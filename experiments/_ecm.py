import copy
import torch
import matplotlib.pyplot as plt
from functorch import vmap
from torch.distributions.normal import Normal
from SOBER._prior import TruncatedGaussian
from SOBER._utils import TensorManager


class CanonicalECMTwoRCs(TensorManager):
    def __init__(self, rt, r1_, t1, r2_, t2, sigma, omega):
        """
        Args:
            - Rt: torch.tensor, the total resistance of the battery (aka R at f=0 [Hz])
            - rt: torch.tensor, the normalised DC resistance
            - ri: torch.tensor, the normalised AC resistance of the i-th RC pair
            - ti: torch.tensor, the normalised time constant of the i-th RC pair
            - sigma: torch.tensor, experimental noise variance
            - omega: torch.tensor, angular frequency [rad/s]
        """
        super().__init__()
        self.omega = self.standardise_tensor(omega)
        self.noise_sig = self.tensor(sigma)
        self.normalise_freq()
        self.set_parameters(rt, r1_, t1, r2_, t2)
        self.synthetic_data(self.noise_sig)

    def set_parameters(self, rt, r1_, t1, r2_, t2):
        """
        Args:
            - rt: torch.tensor, the normalised DC resistance
            - ri: torch.tensor, the normalised AC resistance of the i-th RC pair
            - ti: torch.tensor, the normalised time constant of the i-th RC pair
        """
        self.rt = rt
        self.t1 = t1
        self.r1 = torch.exp(-torch.exp(r1_))
        self.t2 = t2
        self.r2 = torch.exp(-torch.exp(r2_))
        self.Rt = torch.exp(self.rt)
        self.r0 = 1 - self.r1 - self.r2

    def normalise_freq(self):
        self.mu = torch.mean(torch.log(self.omega))
        self.sigma = torch.std(torch.log(self.omega))

    def unnormalise_tau(self, tau):
        """
        Args:
            - tau: torch.tensor, time constant tau in log space; tau = ln(omega * t_i)
        Returns:
            - tau: torch.tensor, time constant tau in raw space; tau = omega * t_i
        """
        return torch.exp(-(self.sigma * tau + self.mu))

    def normalised_input(self, tau):
        """
        Args:
            - tau: torch.tensor, time constant tau in raw space; tau = omega * t_i
        Returns:
            - tau: torch.tensor, time constant tau in log space; tau = ln(omega * t_i)
        """
        return torch.log(self.omega) - (self.sigma * tau + self.mu)

    def real_part(self):
        """
        Returns:
            - Z.real: torch.tensor, real part of impedance spectrum
        """
        return self.Rt * (
            self.r0 + self.r1 / 2 * (1 - torch.tanh(self.normalised_input(self.t1)))
            + self.r2 / 2 * (1 - torch.tanh(self.normalised_input(self.t2)))
        )

    def imarginary_part(self):
        """
        Returns:
            - Z.imaginary: torch.tensor, imaginary part of impedance spectrum
        """
        return self.Rt * (
            (self.r1 / 2) / torch.cosh(self.normalised_input(self.t1))
            + (self.r2 / 2) / torch.cosh(self.normalised_input(self.t2))
        )

    def synthetic_data(self, sigma):
        """
        Args:
            - sigma: torch.tensor, experimental noise variance
        """
        R = torch.exp(-torch.exp(sigma))
        std_norm = Normal(self.tensor(0), self.tensor(1))
        self.reZ = self.real_part() + std_norm.sample(torch.Size([len(self.omega)])) * torch.sqrt(R)
        self.imZ = self.imarginary_part() + std_norm.sample(torch.Size([len(self.omega)])) * torch.sqrt(R)
        
    def error(self, _theta):
        theta = torch.squeeze(_theta).detach()
        self.set_parameters(theta[0], theta[1], theta[2], theta[3], theta[4])
        err_reZ = self.reZ - self.real_part()
        err_imZ = self.imZ - self.imarginary_part()
        err = (err_reZ @ err_reZ + err_imZ @ err_imZ)
        return err

    def loglikelihood(self, err):
        N = 2 * len(self.omega)
        R = err / N
        LL = - 0.5 * torch.log(2 * torch.pi * R) * N - 0.5 * err / R
        return LL
    
    def discrepancy(self, err):
        N = 2 * len(self.omega)
        R = err / N
        return - R.log()

    def convert_circuit_elements(self):
        """
        Returns:
            - R0: torch.tensor, the unnormalised DC resistance
            - Ri: torch.tensor, the unnormalised AC resistance of the i-th RC pair
            - Ci: torch.tensor, the unnormalised capacitance of the i-th RC pair
        """
        R0 = self.Rt * self.r0
        R1 = self.Rt * self.r1
        R2 = self.Rt * self.r2
        lnt1 = torch.exp(-(self.sigma * self.t1 + self.mu))
        C1 = lnt1 / R1
        lnt2 = torch.exp(-(self.sigma * self.t2 + self.mu))
        C2 = lnt2 / R2
        return R0, R1, C1, R2, C2

    def plot(self):
        # without noise
        plt.scatter(
            self.numpy(self.real_part()), 
            self.numpy(self.imarginary_part()),
        )
        plt.show()
        plt.scatter(
            self.numpy(torch.log10(self.omega / (2 * torch.pi))),
            self.numpy(self.real_part()),
        )
        plt.scatter(
            self.numpy(torch.log10(self.omega / (2 * torch.pi))),
            self.numpy(self.imarginary_part()),
        )
        plt.show()

        # with noise
        plt.scatter(
            self.numpy(self.reZ), 
            self.numpy(self.imZ),
        )
        plt.show()
        plt.scatter(
            self.numpy(torch.log10(self.omega / (2 * torch.pi))),
            self.numpy(self.reZ),
        )
        plt.scatter(
            self.numpy(torch.log10(self.omega / (2 * torch.pi))),
            self.numpy(self.imZ),
        )
        plt.show()

    def __call__(self, _theta):
        """
        Args:
            - _theta: torch.tensor, circuit parameters, _theta = [rt, r1_, t1, r2_, t2]
        Returns:
            - LL: torch.tensor, log-likelihood
        """
        err = self.error(_theta)
        d = self.discrepancy(err)
        LL = self.loglikelihood(err)
        return d, LL

def setup_ecm_two():
    """
    Set up the experiments with equivalent circuit model simulation
    
    Return:
    - prior: class, the function of mixed prior
    - TestFunction: class, the function that returns true Ackley function value
    """
    # dataset conditions
    n_data = 100  # number of data points
    f = torch.logspace(1, 10, n_data)  # frequency [Hz]
    omega = 2 * torch.pi * f  # angular frequency [rad/s]
    params_true = torch.tensor([2, -0.5, -1,0, 0.5])  # true parameter set

    # ECM parameters
    rt, r1, t1, r2, t2 = params_true  # decompose the parameters
    sigma = 1.0 # true noise variance
    TwoRCsModel = CanonicalECMTwoRCs(rt, r1, t1, r2, t2, sigma, omega)  # set ECM
    
    mu_pi = params_true * 0.9  # mean vector of Gaussian prior
    cov_pi = 0.5 * torch.diag(torch.ones(len(mu_pi))).float()  # covariance matrix of Gaussian prior
    bounds = torch.tensor([
        [1,-2,-2,-2, -2],
        [3, 2, 2, 2,  2],
    ])
    prior = TruncatedGaussian(mu_pi, cov_pi, bounds)
    
    TestFunction = vmap(TwoRCsModel) # True discrepancy function
    return prior, TestFunction
