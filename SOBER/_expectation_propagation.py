from ._sober_wrapper import SoberWrapper
from ._prior import TruncatedGaussian
import torch


class ExpectationPropagation(SoberWrapper):

    def __init__(
        self,
        model,
        data,
        feature_extractor,
        model_initial_samples=0,
        mean=None,
        covariance=None,
        bounds=None,
        use_bolfi=False,
        transforms=None,
        seed=None,
        disable_numpy_mode=False,
        parallelization=True,
        visualizations=False,
        true_optimum=None,
        **kwargs
    ):
        """!
        :param model:
            A method that takes an array of numbers and returns an array
            of numbers. These are intended to be parameters and model
            evaluation, respectively. This method will provide the
            training data throughout the inverse model training.
        :param data:
            The data that `model` will be fitted to. Has to have the
            same shape as the return value of model`. If not set,
            `custom_objective_and_loglikelihood` needs to be used.
        :param model_initial_samples:
            Number of parameter samples from the prior, that get used as
            initial training data together with their model evaluations.
        :param mean:
            A torch.Tensor of size n, where n is the number of model
            parameters. The mean used for a Gaussian prior. Note that
            this refers to the input of `model` and hence defines the
            target area where the inverse model will be trained.
            This will be internally modified with `transforms`.
        :param covariance:
            A torch.Tensor of size n x n, where n is the number of model
            parameter. The covariance used for a Gaussian prior. Note
            that this refers to the input of `model` and hence defines
            the target area where the inverse model will be trained.
            This will NOT be internally modified with `transforms`.
        :param bounds:
            A torch.Tensor of size 2 x n, where n is the number of model
            parameters. Entry 0/1 is the list of lower/upper bounds.
            This will be internally modified with `transforms`.
        :param use_bolfi:
            By default, use a simple RBF (Radial Basis Function) kernel.
            This describes the surrogate model that will be trained.
            If set to True, the surrogate model will get additional
            structure that prepares it for optimization tasks. It will
            be more inclined to have a parabolic shape, for instance.
            For information, see https://doi.org/10.1002/batt.202200374,
            or for the details, http://jmlr.org/papers/v17/15-017.html.
            Recommended for optimization tasks that would otherwise
            overwhelm the capabilities of the machine this runs on.
        :param transforms:
            An optional list of 2-tuples, where the first entries will
            be used to transform the model parameters before
            parameterization, and the second entries will be used to
            reverse that transformation to get the parameterization
            results out of the calculations done in transformed space.
        :param seed:
            An optional seed to fix Random Number Generation.
        :param disable_numpy_mode:
            When set to True, the cast of parameter sets to NumPy arrays
            will be disabled. Use this to use your torch-compatible
            model in high-performance mode.
        :param parallelization:
            By default, `model` will get called on as many CPUs in
            parallel as the system provides. If set to off, `model` will
            be called on batch input and expected to parallelize itself.
        :param visualizations:
            If set to True, visualizations of the parameterization
            process get plotted alongside it. Note that the plot windows
            need to be closed then for the algorithm to progress.
        :param true_optimum:
            Will be used for visualizations if set. Has to have the same
            shape as one parameter set.
        :param **kwargs:
            Additional keyword arguments will be passed to the `model`.
        """
        super().__init__(
            model,
            data,
            model_initial_samples,
            mean,
            covariance,
            bounds,
            'TruncatedGaussian',  # prior
            False,  # maximize
            use_bolfi,
            None,  # weights
            None,  # custom_objective_and_loglikelihood
            transforms,
            seed,
            disable_numpy_mode,
            parallelization,
            visualizations,
            true_optimum,
            standalone=False,
            **kwargs
        )

        self.normalized_mean = self.prior.mvn.mean
        self.normalized_covariance = self.prior.mvn.covariance_matrix

        self.feature_extractor = feature_extractor
        self.experimental_features = self.feature_extractor(self.data)
        self.feature_dim = len(self.experimental_features)
        self.current_feature = 0

        # These names correspond to the Exponential Family description
        # of a Normal distribution: µ = Q⁻¹ r, Σ = Q⁻¹.
        self.Q = torch.linalg.inv(self.normalized_covariance)
        self.r = self.Q @ self.normalized_mean
        self.Q_features = [
            torch.zeros_like(self.Q) for _ in range(self.feature_dim)
        ]
        self.r_features = [
            torch.zeros_like(self.r) for _ in range(self.feature_dim)
        ]

    def distance_function(self, observations):
        observed_features = [
            self.feature_extractor(single_obs) for single_obs in observations
        ]
        return torch.tensor([
            (
                single_obs_f[self.current_feature]
                - self.experimental_features[self.current_feature]
            ).norm()
            for single_obs_f in observed_features
        ]).to(device=self.tm.device, dtype=self.tm.dtype)

    def run_Expectation_Propagation(
        self,
        ep_iterations=3,
        final_dampening=0.5,
        **kwargs
    ):
        """
        Performs Expectation Propagation with SOBER and BASQ.

        :param ep_iterations:
            The number of times the features shall be looped over.
        :param final_dampening:
            The dampening in the final result. This will be calculated
            down to the dampening applied in each feature update.
        :param **kwargs:
            Give the parameters that ``run_SOBER`` and ``run_BASQ`` will
            use. See there for details. Required.
        """

        ep_dampener = 1 - self.feature_dim * (
            1 - final_dampening**(1 / (self.feature_dim * ep_iterations))
        )

        for _ in range(ep_iterations):
            for i in range(self.feature_dim):
                self.current_feature = i
                self.initialize_sober()
                self.run_SOBER(**kwargs)
                taken_samples, _, _, _, _ = self.run_BASQ(
                    return_raw_samples=True, **kwargs
                )
                # These give the posterior in the case of 0 dampening.
                interim_mean = torch.mean(taken_samples, dim=0)
                interim_covariance = torch.cov(taken_samples.T)
                Q_interim = torch.linalg.inv(interim_covariance)
                r_interim = Q_interim @ interim_mean
                self.Q_features[i] += (1 - ep_dampener) * (Q_interim - self.Q)
                self.r_features[i] += (1 - ep_dampener) * (r_interim - self.r)
                self.Q = (1 - ep_dampener) * Q_interim + ep_dampener * self.Q
                self.r = (1 - ep_dampener) * r_interim + ep_dampener * self.r
                posterior_covariance = torch.linalg.inv(self.Q)
                posterior_mean = posterior_covariance @ self.r
                posterior_bounds = torch.stack([
                    (
                        posterior_mean
                        - 1.95 * torch.sqrt(torch.diag(posterior_covariance))
                    ),
                    (
                        posterior_mean
                        + 1.95 * torch.sqrt(torch.diag(posterior_covariance))
                    ),
                ]).to(device=self.tm.device, dtype=self.tm.dtype)
                self.prior = TruncatedGaussian(
                    posterior_mean,
                    posterior_covariance,
                    posterior_bounds
                )
