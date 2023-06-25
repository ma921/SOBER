import copy
import torch
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal


class CanonicalECMTwoRCs:
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
        self.omega = omega
        self.noise_sig = torch.tensor(sigma)
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
        self.reZ = self.real_part() + Normal(0, 1).sample(torch.Size([len(self.omega)])) * torch.sqrt(R)
        self.imZ = self.imarginary_part() + Normal(0, 1).sample(torch.Size([len(self.omega)])) * torch.sqrt(R)
        self.Rt_syn = copy.deepcopy(self.Rt)
        self.LL = self.loglikelihood(sigma)

    def loglikelihood(self, sigma):
        """
        Args:
            - sigma: torch.tensor, experimental noise variance
        Returns:
            - LL: torch.tensor, log-likelihood
        """
        R = torch.exp(-torch.exp(sigma))
        err_reZ = self.reZ - self.real_part()
        err_imZ = self.imZ - self.imarginary_part()
        err = (err_reZ @ err_reZ + err_imZ @ err_imZ)
        LL = -torch.log(2 * torch.pi * R) * len(self.omega) - 0.5 * err / R
        return LL

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
        plt.scatter(self.real_part(), self.imarginary_part())
        plt.show()
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.real_part())
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.imarginary_part())
        plt.show()

        # with noise
        plt.scatter(self.reZ, self.imZ)
        plt.show()
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.reZ)
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.imZ)
        plt.show()

    def __call__(self, _theta):
        """
        Args:
            - _theta: torch.tensor, circuit parameters, _theta = [rt, r1_, t1, r2_, t2, sigma]
        Returns:
            - LL: torch.tensor, log-likelihood
        """
        theta = torch.squeeze(_theta).detach()
        R = torch.exp(-torch.exp(theta[-1]))
        self.set_parameters(theta[0], theta[1], theta[2], theta[3], theta[4])
        err_reZ = self.reZ - self.real_part()
        err_imZ = self.imZ - self.imarginary_part()
        err = (err_reZ @ err_reZ + err_imZ @ err_imZ)
        LL = -torch.log(2 * torch.pi * R) * len(self.omega) - 0.5 * err / R
        return LL


class CanonicalECMOneRCs:
    def __init__(self, rt, r1_, t1, sigma, omega):
        """
        Rt: the total resistance of the battery (aka R at f=0 [Hz])
        r0: the normalised DC resistance
        ri: the normalised AC resistance of the i-th RC pair
        ti: the normalised time constant of the i-th RC pair
        sigma: experimental noise variance
        omega: angular frequency [rad/s]
        """
        self.omega = omega
        self.noise_sig = torch.tensor(sigma)
        self.normalise_freq()
        self.set_parameters(rt, r1_, t1)
        self.synthetic_data(self.noise_sig)

    def set_parameters(self, rt, r1_, t1):
        self.rt = rt
        self.t1 = t1
        self.r1 = torch.exp(-torch.exp(r1_))
        self.Rt = torch.exp(self.rt)
        self.r0 = 1 - self.r1

    def normalise_freq(self):
        self.mu = torch.mean(torch.log(self.omega))
        self.sigma = torch.std(torch.log(self.omega))

    def unnormalise_tau(self, tau):
        return torch.exp(-(self.sigma * tau + self.mu))

    def normalised_input(self, tau):
        return torch.log(self.omega) - (self.sigma * tau + self.mu)

    def real_part(self):
        return self.Rt * (
            self.r0 + self.r1 / 2 * (1 - torch.tanh(self.normalised_input(self.t1)))
        )

    def imarginary_part(self):
        return self.Rt * (
            (self.r1 / 2) / torch.cosh(self.normalised_input(self.t1))
        )

    def synthetic_data(self, sigma):
        R = torch.exp(-torch.exp(sigma))
        self.reZ = self.real_part() + Normal(0, 1).sample(torch.Size([len(self.omega)])) * torch.sqrt(R)
        self.imZ = self.imarginary_part() + Normal(0, 1).sample(torch.Size([len(self.omega)])) * torch.sqrt(R)

    def set_true_data(self, reZ, imZ):
        self.reZ = reZ
        self.imZ = imZ

    def convert_circuit_elements(self):
        R0 = self.Rt * self.r0
        R1 = self.Rt * self.r1
        lnt1 = torch.exp(-(self.sigma * self.t1 + self.mu))
        C1 = lnt1 / R1
        return R0, R1, C1

    def plot(self):
        # without noise
        plt.scatter(self.real_part(), self.imarginary_part())
        plt.show()
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.real_part())
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.imarginary_part())
        plt.show()

        # with noise
        plt.scatter(self.reZ, self.imZ)
        plt.show()
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.reZ)
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.imZ)
        plt.show()

    def __call__(self, _theta):
        theta = torch.squeeze(_theta)
        R = torch.exp(-torch.exp(theta[-1]))
        self.set_parameters(theta[0], theta[1], theta[2])
        err_reZ = self.reZ - self.real_part()
        err_imZ = self.imZ - self.imarginary_part()
        err = err_reZ @ err_reZ + err_imZ @ err_imZ
        LL = -torch.log(2 * torch.pi * R) * len(self.omega) - 0.5 * err / R
        return LL


class CanonicalECMThreeRCs:
    def __init__(self, rt, r1_, t1, r2_, t2, r3_, t3, sigma, omega):
        """
        Rt: the total resistance of the battery (aka R at f=0 [Hz])
        r0: the normalised DC resistance
        ri: the normalised AC resistance of the i-th RC pair
        ti: the normalised time constant of the i-th RC pair
        sigma: experimental noise variance
        omega: angular frequency [rad/s]
        """
        self.omega = omega
        self.noise_sig = torch.tensor(sigma)
        self.normalise_freq()
        self.set_parameters(rt, r1_, t1, r2_, t2, r3_, t3)
        self.synthetic_data(self.noise_sig)

    def set_parameters(self, rt, r1_, t1, r2_, t2, r3_, t3):
        self.rt = rt
        self.t1 = t1
        self.r1 = torch.exp(-torch.exp(r1_))
        self.t2 = t2
        self.r2 = torch.exp(-torch.exp(r2_))
        self.t3 = t3
        self.r3 = torch.exp(-torch.exp(r3_))
        self.Rt = self.rt
        self.r0 = 1 - self.r1 - self.r2 - self.r3

    def normalise_freq(self):
        self.mu = torch.mean(torch.log(self.omega))
        self.sigma = torch.std(torch.log(self.omega))

    def unnormalise_tau(self, tau):
        return torch.exp(-(self.sigma * tau + self.mu))

    def normalised_input(self, tau):
        return torch.log(self.omega) - (self.sigma * tau + self.mu)

    def real_part(self):
        return self.Rt * (
            self.r0 + self.r1 / 2 * (1 - torch.tanh(self.normalised_input(self.t1)))
            + self.r2 / 2 * (1 - torch.tanh(self.normalised_input(self.t2)))
            + self.r3 / 2 * (1 - torch.tanh(self.normalised_input(self.t3)))
        )

    def imarginary_part(self):
        return self.Rt * (
            (self.r1 / 2) / torch.cosh(self.normalised_input(self.t1))
            + (self.r2 / 2) / torch.cosh(self.normalised_input(self.t2))
            + (self.r3 / 2) / torch.cosh(self.normalised_input(self.t3))
        )

    def synthetic_data(self, sigma):
        R = torch.exp(-torch.exp(sigma))
        self.reZ = self.real_part() + Normal(0, 1).sample(torch.Size([len(self.omega)])) * torch.sqrt(R)
        self.imZ = self.imarginary_part() + Normal(0, 1).sample(torch.Size([len(self.omega)])) * torch.sqrt(R)

    def set_true_data(self, reZ, imZ):
        self.reZ = reZ
        self.imZ = imZ

    def convert_circuit_elements(self):
        R0 = self.Rt * self.r0
        R1 = self.Rt * self.r1
        R2 = self.Rt * self.r2
        R3 = self.Rt * self.r3
        lnt1 = torch.exp(-(self.sigma * self.t1 + self.mu))
        C1 = lnt1 / R1
        lnt2 = torch.exp(-(self.sigma * self.t2 + self.mu))
        C2 = lnt2 / R2
        lnt3 = torch.exp(-(self.sigma * self.t3 + self.mu))
        C3 = lnt3 / R3
        return R0, R1, C1, R2, C2, R3, C3

    def plot(self):
        # without noise
        plt.scatter(self.real_part(), self.imarginary_part())
        plt.show()
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.real_part())
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.imarginary_part())
        plt.show()

        # with noise
        plt.scatter(self.reZ, self.imZ)
        plt.show()
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.reZ)
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.imZ)
        plt.show()

    def __call__(self, _theta):
        theta = torch.squeeze(_theta)
        R = torch.exp(-torch.exp(theta[-1]))
        self.set_parameters(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6])
        err_reZ = self.reZ - self.real_part()
        err_imZ = self.imZ - self.imarginary_part()
        err = err_reZ @ err_reZ + err_imZ @ err_imZ
        LL = -torch.log(2 * torch.pi * R) * len(self.omega) - 0.5 * err / R
        return LL
    
class CanonicalECMFiveRCs:
    def __init__(self, rt, r1_, t1, r2_, t2, r3_, t3, r4_, t4, r5_, t5, sigma, omega):
        """
        Rt: the total resistance of the battery (aka R at f=0 [Hz])
        r0: the normalised DC resistance
        ri: the normalised AC resistance of the i-th RC pair
        ti: the normalised time constant of the i-th RC pair
        sigma: experimental noise variance
        omega: angular frequency [rad/s]
        """
        self.omega = omega
        self.noise_sig = torch.tensor(sigma)
        self.normalise_freq()
        self.set_parameters(rt, r1_, t1, r2_, t2, r3_, t3, r4_, t4, r5_, t5)
        self.synthetic_data(self.noise_sig)

    def set_parameters(self, rt, r1_, t1, r2_, t2, r3_, t3, r4_, t4, r5_, t5):
        self.rt = rt
        self.t1 = t1
        self.r1 = torch.exp(-torch.exp(r1_))
        self.t2 = t2
        self.r2 = torch.exp(-torch.exp(r2_))
        self.t3 = t3
        self.r3 = torch.exp(-torch.exp(r3_))
        self.t4 = t4
        self.r4 = torch.exp(-torch.exp(r4_))
        self.t5 = t5
        self.r5 = torch.exp(-torch.exp(r5_))
        self.Rt = self.rt
        self.r0 = 1 - self.r1 - self.r2 - self.r3 - self.r4 - self.r5

    def normalise_freq(self):
        self.mu = torch.mean(torch.log(self.omega))
        self.sigma = torch.std(torch.log(self.omega))

    def unnormalise_tau(self, tau):
        return torch.exp(-(self.sigma * tau + self.mu))

    def normalised_input(self, tau):
        return torch.log(self.omega) - (self.sigma * tau + self.mu)

    def real_part(self):
        return self.Rt * (
            self.r0 + self.r1 / 2 * (1 - torch.tanh(self.normalised_input(self.t1)))
            + self.r2 / 2 * (1 - torch.tanh(self.normalised_input(self.t2)))
            + self.r3 / 2 * (1 - torch.tanh(self.normalised_input(self.t3)))
            + self.r4 / 2 * (1 - torch.tanh(self.normalised_input(self.t4)))
            + self.r5 / 2 * (1 - torch.tanh(self.normalised_input(self.t5)))
        )

    def imarginary_part(self):
        return self.Rt * (
            (self.r1 / 2) / torch.cosh(self.normalised_input(self.t1))
            + (self.r2 / 2) / torch.cosh(self.normalised_input(self.t2))
            + (self.r3 / 2) / torch.cosh(self.normalised_input(self.t3))
            + (self.r4 / 2) / torch.cosh(self.normalised_input(self.t4))
            + (self.r5 / 2) / torch.cosh(self.normalised_input(self.t5))
        )

    def synthetic_data(self, sigma):
        R = torch.exp(-torch.exp(sigma))
        self.reZ = self.real_part() + Normal(0, 1).sample(torch.Size([len(self.omega)])) * torch.sqrt(R)
        self.imZ = self.imarginary_part() + Normal(0, 1).sample(torch.Size([len(self.omega)])) * torch.sqrt(R)

    def set_true_data(self, reZ, imZ):
        self.reZ = reZ
        self.imZ = imZ

    def convert_circuit_elements(self):
        R0 = self.Rt * self.r0
        R1 = self.Rt * self.r1
        R2 = self.Rt * self.r2
        R3 = self.Rt * self.r3
        R4 = self.Rt * self.r4
        R5 = self.Rt * self.r5
        lnt1 = torch.exp(-(self.sigma * self.t1 + self.mu))
        C1 = lnt1 / R1
        lnt2 = torch.exp(-(self.sigma * self.t2 + self.mu))
        C2 = lnt2 / R2
        lnt3 = torch.exp(-(self.sigma * self.t3 + self.mu))
        C3 = lnt3 / R3
        lnt4 = torch.exp(-(self.sigma * self.t4 + self.mu))
        C4 = lnt4 / R4
        lnt5 = torch.exp(-(self.sigma * self.t5 + self.mu))
        C5 = lnt5 / R5
        return R0, R1, C1, R2, C2, R3, C3, R4, C4, R5, C5

    def plot(self):
        # without noise
        plt.scatter(self.real_part(), self.imarginary_part())
        plt.show()
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.real_part())
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.imarginary_part())
        plt.show()

        # with noise
        plt.scatter(self.reZ, self.imZ)
        plt.show()
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.reZ)
        plt.scatter(torch.log10(self.omega / (2 * torch.pi)), self.imZ)
        plt.show()

    def __call__(self, _theta):
        theta = torch.squeeze(_theta)
        R = torch.exp(-torch.exp(theta[-1]))
        self.set_parameters(theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8], theta[9], theta[10])
        err_reZ = self.reZ - self.real_part()
        err_imZ = self.imZ - self.imarginary_part()
        err = err_reZ @ err_reZ + err_imZ @ err_imZ
        LL = -torch.log(2 * torch.pi * R) * len(self.omega) - 0.5 * err / R
        return LL
