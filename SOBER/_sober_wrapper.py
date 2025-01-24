from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from copy import deepcopy
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from math import exp, log
from mecbo.BOLFI.botorch_acquisition import SOBERUCB
from mecbo.BOLFI.gpytorch_bolfi_model import BOLFIModel
from multiprocessing import Pool
from scipy.stats import chi2
from ._prior import Uniform, Gaussian, TruncatedGaussian
from ._sober import Sober
from ._utils import TensorManager
from BASQ._basq import BASQ
from BASQ._scale_mmlt import ScaleMmltGP
import time
import torch
import warnings


class SoberWrapper:

    def __init__(
        self,
        model=None,
        data=None,
        model_initial_samples=0,
        mean=None,
        covariance=None,
        bounds=None,
        prior='Uniform',
        maximize=False,
        use_bolfi=False,
        weights=None,
        custom_objective_and_loglikelihood=None,
        transforms=None,
        seed=None,
        disable_numpy_mode=False,
        parallelization=True,
        visualizations=False,
        true_optimum=None,
        standalone=True,
        **kwargs
    ):
        """
        :param model:
            A method that takes an array of numbers and returns an array
            of numbers. These are intended to be parameters and
            model evaluation, respectively. If not set,
            `custom_objective_and_loglikelihood` needs to be used.
        :param data:
            The data that `model` will be fitted to. Has to have the
            same shape as the return value of `model`. If not set,
            `custom_objective_and_loglikelihood` needs to be used.
        :param model_initial_samples:
            Number of parameter samples from the prior, that get used as
            initial training data together with their model evaluations.
        :param mean:
            A torch.Tensor of size n, where n is the number of model
            parameters. The mean used for a Gaussian prior.
            This will be internally modified with `transforms`.
        :param covariance:
            A torch.Tensor of size n x n, where n is the number of model
            parameter. The covariance used for a Gaussian prior.
            This will NOT be internally modified with `transforms`.
        :param bounds:
            A torch.Tensor of size 2 x n, where n is the number of model
            parameters. Entry 0/1 is the list of lower/upper bounds.
            This will be internally modified with `transforms`.
        :param prior:
            Defaults to the 'Uniform' prior within `bounds`.
            May be set to 'Gaussian' using `mean` and `covariance` or
            'TruncatedGaussian' using all three.
        :param maximize:
            If set to True, the model distance to data will be maximized
            instead of minimized. Has no effect with custom objectives.
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
        :param weights:
            An optional list of weights that will be applied to the
            distance between `model` and `data`. Defaults to 1.0.
        :param custom_objective_and_loglikelihood:
            To use SOBER directly through this wrapper, set this to a
            function that returns a two-tuple, given the input of
            'model'. First entry of the tuple is the objective value
            that will be maximized, and second entry is the
            Log-Likelihood.
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
        :param standalone:
            By default, when initializing an object of this class
            directly, this is True and will initialize SOBER.
            Will be set to False by inheriting classes if necessary.
        :param **kwargs:
            Additional keyword arguments will be passed to the `model`.
        """
        if visualizations:
            import matplotlib.pyplot as plt
            from pandas import DataFrame
            from seaborn import pairplot

        self.tm = TensorManager()

        self.model = model
        self.model_kwargs = kwargs
        self.data = data

        if bounds is not None:
            self.input_dim = len(bounds[0])
        elif mean is not None:
            self.input_dim = len(mean)
        else:
            raise ValueError(
                "Either 'mean' and 'covariance' or 'bounds' needs to be set."
            )
        self.transforms = (
            transforms or [(None, None) for _ in range(self.input_dim)]
        )
        for i in range(len(self.transforms)):
            if not self.transforms[i][0] or not self.transforms[i][1]:
                self.transforms[i] = (lambda x: x, lambda x: x)
        if mean is not None:
            self.mean = mean
            batched_and_transformed_mean = self.apply_transform(
                torch.atleast_2d(deepcopy(mean))
            )

        if bounds is not None:
            self.bounds = bounds.to(device=self.tm.device, dtype=self.tm.dtype)
            self.bounds[0] = self.apply_transform(
                torch.atleast_2d(self.bounds[0])
            )[0]
            self.bounds[1] = self.apply_transform(
                torch.atleast_2d(self.bounds[1])
            )[0]
            if mean is None:
                self.mean = self.reverse_transform(torch.atleast_2d(
                    (self.bounds[0] + self.bounds[1]) / 2
                ))
        elif mean is not None and covariance is not None:
            self.bounds = torch.stack([
                (
                    batched_and_transformed_mean
                    - 4 * torch.sqrt(torch.diag(covariance))
                ),
                (
                    batched_and_transformed_mean
                    + 4 * torch.sqrt(torch.diag(covariance))
                ),
            ]).to(device=self.tm.device, dtype=self.tm.dtype)
        else:
            raise ValueError(
                "Either 'mean' and 'covariance' or 'bounds' needs to be set."
            )

        if 'Gaussian' in prior:
            if covariance is None:
                if bounds is None:
                    raise ValueError(
                        "Either 'covariance' or 'bounds' needs to be set."
                    )
                covariance = torch.diag(
                    (self.bounds[1] - self.bounds[0])
                    / (2 * chi2(self.input_dim).ppf(0.95)**0.5)
                )

        if prior == 'Uniform':
            self.diagonalization = torch.diag(torch.ones(self.input_dim))
            self.diagonalization = self.diagonalization.to(
                device=self.tm.device, dtype=self.tm.dtype
            )
            self.prior = Uniform(torch.stack([
                torch.zeros(self.input_dim), torch.ones(self.input_dim)
            ]))
        elif prior == 'Gaussian':
            _, self.diagonalization = torch.linalg.eigh(covariance)
            self.prior = Gaussian(
                self.normalize_input(batched_and_transformed_mean)[0],
                (0.5 / 4)**2 * torch.diag(torch.ones(self.input_dim))
            )
        elif prior == 'TruncatedGaussian':
            _, self.diagonalization = torch.linalg.eigh(covariance)
            self.diagonalization = self.diagonalization.to(
                device=self.tm.device, dtype=self.tm.dtype
            )
            self.prior = TruncatedGaussian(
                self.normalize_input(batched_and_transformed_mean)[0],
                (0.5 / 4)**2 * torch.diag(torch.ones(self.input_dim)),
                torch.stack([
                    torch.zeros(self.input_dim), torch.ones(self.input_dim)
                ])
            )
        else:
            raise ValueError(
                "'prior' must be one of 'Uniform', 'Gaussian', or "
                "'TruncatedGaussian'."
            )
        self.back_diagonalization = self.diagonalization.T

        # Match eigenvalue order with parameter order.
        # This will be only used for visualization, and even then it
        # is only correct in the case of a diagonal prior covariance.
        self.diag_order = [-1 for _ in range(self.input_dim)]
        for i in range(self.input_dim):
            result_orig = self.normalize_input(
                self.apply_transform(torch.atleast_2d(deepcopy(self.mean)))
            )
            test_vector = self.apply_transform(
                torch.atleast_2d(deepcopy(self.mean))
            )
            test_vector[0][i] = self.bounds[0][i]
            result_eval = self.normalize_input(test_vector)
            comparison = (result_orig - result_eval).abs()[0]
            self.diag_order[i] = int(comparison.argmax())
        self.current_MAP = self.mean

        self.maximize = maximize

        self.use_bolfi = use_bolfi
        if weights is None and data is not None:
            self.weights = 1.0
        else:
            self.weights = weights
        self.custom_objective_and_loglikelihood = (
            custom_objective_and_loglikelihood
        )
        self.disable_numpy_mode = disable_numpy_mode
        self.parallelization = parallelization

        if seed:
            torch.manual_seed(seed)

        self.true_optimum = true_optimum
        if self.true_optimum is not None:
            self.normalized_true_optimum = self.normalize_input(
                self.apply_transform(
                    torch.atleast_2d(torch.Tensor(self.true_optimum))
                )
            )[0]
        else:
            self.normalized_true_optimum = None

        self.X_all = self.prior.sample(model_initial_samples).to(
            device=self.tm.device, dtype=self.tm.dtype
        )
        if visualizations:
            pairgrid = pairplot(DataFrame(self.tm.numpy(self.X_all)))
            pairgrid.figure.suptitle("correlation plot of prior sampling")
            if self.normalized_true_optimum is not None:
                for i in range(len(self.true_optimum)):
                    pairgrid.axes[i][i].scatter(
                        self.normalized_true_optimum[i],
                        0, s=100, marker='*', color='C1'
                    )
            plt.show()

        # To store and then check against the MAP samples
        self.sober_iterations = 0
        self.surrogate_effective_samples = 0

        self.standalone = standalone
        if self.standalone:
            self.initialize_sober(visualizations)

    def initialize_sober(self, visualizations=False):
        if visualizations:
            import matplotlib.pyplot as plt

        self.Y_all, self.LL_all = self.objective_and_loglikelihood_function(
            self.X_all, sober_batch=True
        )
        # Normalize Y_all and store the normalization parameters.
        # For applying Y_batch later, de-normalization is needed.
        self.Y_all_mean = self.Y_all.mean()
        self.Y_all_std = self.Y_all.std()
        self.Y_all = (self.Y_all - self.Y_all_mean) / self.Y_all_std

        if visualizations:
            _, ax = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4))
            ax[0].hist(self.tm.numpy(
                self.Y_all_mean + self.Y_all_std * self.Y_all
            ), 50)
            if self.custom_objective_and_loglikelihood is None:
                ax[0].set_title("log distances histogram")
                ax[0].set_xlabel("log distance values")
            else:
                ax[0].set_title("custom objective histogram")
                ax[0].set_xlabel("custom objective values")
            ax[0].set_ylabel("occurrences")
            ax[1].hist(self.tm.numpy(self.LL_all), 50)
            ax[1].set_title("log likelihoods histogram")
            ax[1].set_xlabel("log likelihood values")
            plt.show()

        # Sets self.surrogate_model.
        self.set_rbf_model(self.X_all, self.Y_all, use_bolfi=self.use_bolfi)
        self.sober = Sober(self.prior, self.surrogate_model)
        self.results = []
        self.total_sober_iterations = 0
        self.total_model_samples = []

    def process_evaluations(self, evaluations, sober_batch):
        """
        Gets called after model evaluation if standalone is False.

        :param evaulations:
            Evaluations of ``self.model``.
        :param sober_batch: bool
            Used to mark whether the evaluations to be processed are
            part of a batch or not. If not, usually skip processing.
        """
        pass

    def normalize_input(self, x):
        """
        Normalizes transformed parameter values to the unit cube.

        :param x:
            An array of numbers that are valid model parameters.
        :returns:
            An array of numbers that are between 0 and 1.
        """
        return torch.matmul(
            self.diagonalization,
            (
                (x - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
            )[..., None]
        ).squeeze(2)

    def denormalize_input(self, x):
        """
        Maps the unit cube onto the transformed parameter space.

        :param x:
            An array of numbers that are between 0 and 1.
        :returns:
            An array of numbers inside the parameter bounds.
        """
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * (
            torch.matmul(self.back_diagonalization, x[..., None])
        ).squeeze(2)

    def apply_transform(self, x):
        """
        Transforms from parameter space to transformed space.

        :param x:
            An array of numbers from parameter space.
        :returns:
            An array of numbers, transformed with ``self.transforms``.
        """
        if x.dim() > 1:
            for i, transform in enumerate(self.transforms):
                x.T[i] = transform[0](x.T[i])
        else:
            for i, transform in enumerate(self.transforms):
                x[i] = transform[0](x[i])
        return x

    def reverse_transform(self, x):
        """
        Reverses transformed space into parameter space.

        :param x:
            An array of numbers from transformed space.
        :returns:
            An array of numbers that are valid model parameters.
        """
        if x.dim() > 1:
            for i, transform in enumerate(self.transforms):
                x.T[i] = transform[1](x.T[i])
        else:
            for i, transform in enumerate(self.transforms):
                x[i] = transform[1](x[i])
        return x

    def apply_transform_and_normalize_one_variable(self, var, index):
        """
        Translates one variable from parameter space to unit cube.

        :param var:
            The value of the variable.
        :param index:
            The index of the variable.
        :returns:
            A Python float in the unit cube.
        """
        x = deepcopy(self.current_MAP)
        x[index] = var
        return float(self.normalize_input(self.apply_transform(
            torch.atleast_2d(x)
        ))[0][self.diag_order[index]])

    def denormalize_and_reverse_transform_one_variable(self, var, index):
        """
        Translates one variable from unit cube to parameter space.

        :param var:
            The value of the variable.
        :param index:
            The index of the variable.
        :returns:
            A Python float in parameter space.
        """
        x = deepcopy(self.current_MAP)
        x = self.normalize_input(self.apply_transform(torch.atleast_2d(x)))[0]
        x[self.diag_order[index]] = var
        return float(self.reverse_transform(self.denormalize_input(
            torch.atleast_2d(x)
        ))[0][index])

    def model_wrapper(self, x):
        """
        Calls ``self.model``, with or without NumPy conversion.

        :param x:
            An array of numbers that are valid model parameters.
        :returns:
            A torch.Tensor of the model results.
        """
        if self.disable_numpy_mode:
            return self.model(x, **self.model_kwargs)
        else:
            return torch.Tensor(self.model(
                x.cpu().detach().numpy(), **self.model_kwargs
            )).to(device=self.tm.device, dtype=self.tm.dtype)

    @staticmethod
    def parallelizable_model_wrapper(
        x, model, disable_numpy_mode, device, dtype, model_kwargs
    ):
        if disable_numpy_mode:
            return model(x, **model_kwargs)
        else:
            result = torch.tensor(
                model(x.cpu().detach().numpy(), **model_kwargs)
            )
            if result.dtype == torch.cfloat or result.dtype == torch.cdouble:
                return result.to(device=device)
            else:
                return result.to(device=device, dtype=dtype)

    def distance_function(self, observations):
        """
        The distance between ``self.data`` and model evaluation.

        :param observations:
            A ``self.model`` evaulation.
        :returns:
            The 2-norm weighted distance between model and data.
        """
        return (
            (observations - self.data) * self.weights
        ).view(observations.shape[0], -1).norm(dim=1).to(
            device=self.tm.device, dtype=self.tm.dtype
        )

    def default_objective_function(self, observations):
        """
        Takes the negative logarithm of distance for maximization.

        :param observations:
            A ``self.model`` evaluation.
        :returns:
            The negative of the logarithm of ``self.distance_function``.
        """
        if isinstance(observations, list):
            try:
                observations = torch.stack(observations)
            except RuntimeError:  # inhomogeneous observation shape
                return torch.tensor([
                    -self.distance_function(obs.unsqueeze(0)).log()
                    for obs in observations
                ]).to(device=self.tm.device, dtype=self.tm.dtype)
        return -self.distance_function(observations).log()

    def evaluate_model(self, x):
        """
        Calls the model in parallel and produces batch output.

        :param x:
            An array of numbers after transformation and normalization.
            May be batched, where the batch-dimension is the first one.
        :returns:
            An array of cost function values at x.
        """
        randomness = 'different'
        transformed_batch = self.denormalize_input(torch.atleast_2d(x))
        # torch.vmap works with in-place operations only.
        batch = torch.vmap(
            self.reverse_transform, randomness=randomness
        )(transformed_batch)
        # Since this calls the model, use multiprocessing instead.
        if self.parallelization:
            parallel_input = [
                (
                    batch_entry,
                    self.model,
                    self.disable_numpy_mode,
                    self.tm.device,
                    self.tm.dtype,
                    self.model_kwargs,
                )
                for batch_entry in batch
            ]
            try:
                with Pool() as p:
                    evaluations = p.starmap(
                        SoberWrapper.parallelizable_model_wrapper,
                        parallel_input
                    )
            except AttributeError as e:
                raise AttributeError(
                    "The 'model' must be defined in a global scope, else "
                    "calculating multiple instances in parallel can't work. "
                    "Original error message: " + str(e)
                )
        else:
            evaluations = SoberWrapper.parallelizable_model_wrapper(
                batch,
                self.model,
                self.disable_numpy_mode,
                self.tm.device,
                self.tm.dtype,
                self.model_kwargs
            )
        return evaluations

    def objective_and_loglikelihood_function(self, x, sober_batch=True):
        """
        Simplest choice for a Log-Likelihood, basically a rescaling.

        :param x:
            An array of numbers after transformation and normalization.
            May be batched, where the batch-dimension is the first one.
        :param sober_batch:
            Gets passed to ``self.process_evaluations``.
        :returns:
            2-tuple: an array of objective values at x, and an array of
            Log-Likelihood values at x.
        """

        if self.custom_objective_and_loglikelihood is not None:
            randomness = 'different'
            transformed_batch = self.denormalize_input(torch.atleast_2d(x))
            # torch.vmap works with in-place operations only.
            batch = torch.vmap(
                self.reverse_transform, randomness=randomness
            )(transformed_batch)
            return self.custom_objective_and_loglikelihood(batch)
        evaluations = self.evaluate_model(x)
        if not self.standalone:
            self.process_evaluations(evaluations, sober_batch)
        N = self.input_dim
        objective = self.default_objective_function(evaluations)
        if self.maximize:
            objective = -objective
        loglikelihood = -0.5 * (1 + log(2 * torch.pi / N) - objective) * N
        return objective, loglikelihood

    def set_rbf_model(self, x, y, use_bolfi=False):
        """
        Sets up a simple RBF or advanced BOLFI surrogate.

        :param x:
            The independent variables.
        :param y:
            The dependent variables.
        :param use_bolfi:
            Whether or not to use the BOLFI surrogate. Recommended for
            optimization tasks that would otherwise overwhelm the PC.
        """
        base_kernel = RBFKernel(ard_num_dims=x.shape[-1])
        covar_module = ScaleKernel(base_kernel)
        train_y = y.view(-1).unsqueeze(1)
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-2, 10))
        if use_bolfi:
            surrogate_model = BOLFIModel(
                x, train_y, likelihood=likelihood, bounds=self.bounds
            )
        else:
            surrogate_model = SingleTaskGP(
                x, train_y, likelihood=likelihood, covar_module=covar_module
            )
        if self.tm.is_cuda():
            self.surrogate_model = surrogate_model.cuda()
        else:
            self.surrogate_model = surrogate_model

    def optimize_model(self):
        """Trains the (RBF) surrogate model on the new data."""
        self.surrogate_model.train()
        self.surrogate_model.likelihood.train()
        self.surrogate_model.set_train_data(
            self.X_all, self.Y_all, strict=False
        )
        mll = ExactMarginalLogLikelihood(
            self.surrogate_model.likelihood, self.surrogate_model
        )
        fit_gpytorch_mll(mll)
        self.surrogate_model.eval()
        self.surrogate_model.likelihood.eval()

    def visualize_results(self):
        """Visualizes convergence of SOBER (call 'run' first)."""
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 2, tight_layout=True, figsize=(8, 4))
        ax[0].plot(
            self.total_model_samples,
            [entry[1] for entry in self.results],
            'bo-',
            label="observed maximum"
        )
        ax[0].legend()
        ax[0].set_xlabel("index of batches")
        ax[0].set_ylabel("objective")
        ax[1].plot(
            self.total_model_samples,
            [entry[0] for entry in self.results],
            'bo-'
        )
        ax[1].set_xlabel("index of batches")
        ax[1].set_ylabel("overhead [s]")
        plt.show()

    def results_to_dict(self):
        """Collects results in basic Python objects (lists, dicts)."""
        return {
            "parameters evaluations": [
                list(entry) for entry in self.X_all.cpu().detach().numpy()
            ],
            "objective evaluations": list((
                self.Y_all_mean + self.Y_all_std * self.Y_all
            ).cpu().detach().numpy()),
            "Log-Likelihood evaluations": list(
                self.LL_all.cpu().detach().numpy()
            ),
            "results": {
                "duration [s]": [res[0] for res in self.results],
                "best observed": [res[1] for res in self.results],
            },
        }

    def run_SOBER(
        self,
        sober_iterations,
        model_samples_per_iteration,
        surrogate_samples=None,
        surrogate_effective_samples=None,
        acquisition_function=None,
        visualizations=False,
        verbose=True,
        **kwargs
    ):
        """
        Calls the basic SOBER routines in order.

        :param sober_iterations:
            The number of SOBER loops. In each loop,
            `model_samples_per_iteration` samples are taken from the
            current SOBER surrogate. At the end of each loop, this
            surrogate gets re-trained with those new samples.
        :param model_samples_per_iteration:
            The batch size, i.e., how many samples get taken from the
            current SOBER surrogate model before the surrogate gets
            updated by the new samples (with the model evaluations).
        :param surrogate_samples:
            Number of random samples taken from the surrogate at the
            beginning of each loop. These form the basis for a
            surrogate-surrogate, which will be used to select the
            model evaluation points. This number can be very high, as
            the surrogate is very performant; the reason for the
            surrogate-surrogate is that it's tractable for the
            non-random model sample selection calculations.
            Defaults to 4 times `model_samples_per_iteration`.
        :param surrogate_effective_samples:
            The number of effective samples the surrogate gets reduced
            to before it gets used for sample selection calculations.
            This is intended to reduce the internal complexity of the
            surrogate-surrogate, as its number of "kernels" would
            correspond to `surrogate_samples` elsewise. Usually, the
            number of samples needed to cover the parameter space is way
            higher than the number of "kernels" a surrogate-surrogate
            needs. So to speed up calculations, the surrogate-surrogate
            gets reduced to this number of "kernels".
            Defaults to 2 times `model_samples_per_iteration`.
        :param acquisition_function:
            The heuristic tacked on to the sample selection
            calculations. If None, but use_bolfi=True at initialization,
            the Upper Confidence Bound (UCB) acquisition function will
            be set up and used, as it is part of the BOLFI formula.
            Note: the original BOLFI minimizes, so it uses Lower CB.
        :param visualizations:
            If set to True, visualizations of the parameterization
            process get plotted alongside it. Note that the plot windows
            need to be closed then for the algorithm to progress.
        :param verbose:
            If set to True, status updates of the parameterization get
            logged to stdout.
        :param kwargs:
            Please ignore; it is an implementation detail that makes
            ``run_SOBER_adaptively`` easier to maintain.
        """
        surrogate_effective_samples = (
            surrogate_effective_samples or 2 * model_samples_per_iteration
        )
        if model_samples_per_iteration >= surrogate_effective_samples:
            raise ValueError(
                "Number of model evaluations must be lower than number "
                "of surrogate evaluations."
            )
        surrogate_samples = (
            surrogate_samples or 4 * model_samples_per_iteration
        )
        for _ in range(1, sober_iterations + 1):
            self.sober_iterations += 1
            time_start = time.monotonic()
            self.optimize_model()
            # For some reason, calling this too early would not get rid
            # of the superfluous "input matches training data" error.
            warnings.simplefilter("ignore")
            self.sober.update_model(self.surrogate_model)
            if acquisition_function is None and self.use_bolfi:
                acquisition_function = SOBERUCB(
                    self.surrogate_model, sample_size=len(self.X_all)
                )
            X_batch = self.sober.next_batch(
                surrogate_samples,
                surrogate_effective_samples,
                model_samples_per_iteration,
                calc_obj=acquisition_function,
                verbose=verbose
            )
            self.surrogate_effective_samples = surrogate_effective_samples
            time_end = time.monotonic()
            time_interval = time_end - time_start
            self.X_all = torch.cat((self.X_all, X_batch), dim=0)
            Y_batch, LL_batch = self.objective_and_loglikelihood_function(
                X_batch, sober_batch=True
            )
            # De-normalize Y_all before mixing it with Y_batch.
            self.Y_all = self.Y_all_mean + self.Y_all_std * self.Y_all
            self.Y_all = torch.cat((self.Y_all, Y_batch), dim=0)
            # Now normalize the new Y_all.
            self.Y_all_mean = self.Y_all.mean()
            self.Y_all_std = self.Y_all.std()
            self.Y_all = (self.Y_all - self.Y_all_mean) / self.Y_all_std
            self.LL_all = torch.cat((self.LL_all, LL_batch), dim=0)
            Y_all_denorm = self.Y_all_mean + self.Y_all_std * self.Y_all
            if verbose:
                print(
                    f"{len(self.X_all)}) "
                    f"Best objective: {Y_all_denorm.max().item():.5e} "
                    f"Best Log-Likelihood: {self.LL_all.max().item():.5e}"
                )
                mspersample = time_interval / model_samples_per_iteration * 1e3
                print(
                    f"Acquisition time [s]: {time_interval:.5e}, "
                    f"per sample [ms]: {mspersample:.5e}"
                )
            self.results.append([time_interval, Y_all_denorm.max().item()])
            self.total_sober_iterations += 1
            if self.total_model_samples:
                self.total_model_samples.append(
                    self.total_model_samples[-1] + model_samples_per_iteration
                )
            else:
                self.total_model_samples.append(model_samples_per_iteration)

        if visualizations:
            self.visualize_results()

    def run_BASQ(
        self,
        integration_nodes,
        basq_samples=None,
        basq_effective_samples=None,
        basq_posterior_samples=None,
        map_samples=None,
        dampening=0,
        visualizations=False,
        return_raw_samples=False,
        verbose=True,
        **kwargs
    ):
        """
        Performs BASQ (use 'run_SOBER' first).

        :param integration_nodes:
            Number of nodes for the numerical integration, i.e.,
            quadrature. These will be optimally chosen. May be thought
            of as the BASQ equivalent of the model samples in SOBER.
        :param basq_samples:
            Number of random samples taken from the BASQ model as the
            basis for the computation of the integration nodes.
            Defaults to 4 times `integration_nodes`.
        :param basq_effective_samples:
            The number of effective samples the `basq_samples`
            evaluations get reduced to. These will form the set of nodes
            to choose from for the integration nodes.
            Defaults to 2 times `integration_nodes`.
        :param basq_posterior_samples:
            Number of samples to draw from the BASQ posterior for
            visualization purposes. Defaults to `integration_nodes`.
        :param ratio_wkde:
            The dampening factor used when sampling the BASQ posterior.
            Defaults to 0, i.e., only the posterior. Any value between
            this and 1 will include that fraction of the prior.
        :param map_samples:
            Number of samples to draw from the BASQ posterior in search
            of the Maximum A Posteriori location (MAP). Defaults to the
            'surrogate_effective_samples' used in the SOBER call,
            multiplied by the total amount of SOBER iterations.
        :param visualizations:
            If set to True, visualizations of the parameterization
            process get plotted alongside it. Note that the plot windows
            need to be closed then for the algorithm to progress.
        :param return_raw_samples:
            Set to True if you want to get the samples taken from the
            normalized posterior. Else, parameter values will be given.
        :param verbose:
            If set to True, status updates of the quadrature get logged
            to stdout.
        :param **kwargs:
            Please ignore; it is an implementation detail that makes
            ``run_SOBER_adaptively`` easier to maintain.
        :returns:
            A 5-tuple of the posterior samples, the MAP, the best
            observed model sample, the expected log marginal likelihood,
            and the variance of the log marginal likelihood.
        """
        map_samples = map_samples or (
            self.sober_iterations * self.surrogate_effective_samples
        )
        if map_samples < self.surrogate_effective_samples:
            raise ValueError(
                "Number of MAP samples must be higher than number of "
                "surrogate effective samples."
            )
        basq_samples = basq_samples or 4 * integration_nodes
        basq_effective_samples = (
            basq_effective_samples or 2 * integration_nodes
        )
        basq_posterior_samples = basq_posterior_samples or integration_nodes
        if visualizations:
            import matplotlib.pyplot as plt
            from pandas import DataFrame
            from seaborn import pairplot
        if verbose:
            from tabulate import tabulate

        time_start = time.monotonic()
        basq_base_kernel = RBFKernel(ard_num_dims=self.X_all.shape[-1])
        basq_covar_module = ScaleKernel(basq_base_kernel)
        basq_surrogate_model = ScaleMmltGP(
            self.X_all, self.LL_all, basq_covar_module
        )
        time_setup = time.monotonic()
        basq = BASQ(
            self.prior,
            basq_surrogate_model,
            self.sober,
            ratio_wkde=1 - dampening,
        )
        time_init = time.monotonic()
        (
            log_expected_marginal_likelihood,
            log_approx_variance_marginal_likelihood
        ) = basq.quadrature(
            basq_samples, basq_effective_samples, integration_nodes
        )
        time_quad = time.monotonic()
        taken_samples = basq.sampling_posterior(basq_posterior_samples)
        time_samples = time.monotonic()

        MAP_normalized = basq.MAP(map_samples)
        time_map = time.monotonic()
        if verbose:
            print(
                "BASQ: setup", time_setup - time_start,
                "init", time_init - time_setup,
                "quad", time_quad - time_init,
                "samples", time_samples - time_quad,
                "MAP", time_map - time_samples,
            )
        MAP = self.reverse_transform(self.denormalize_input(
            torch.atleast_2d(deepcopy(MAP_normalized))
        )[0])
        self.current_MAP = MAP
        best_observed_normalized = self.X_all[(
            self.Y_all_mean + self.Y_all_std * self.Y_all
        ).argmax()]
        best_observed = self.reverse_transform(self.denormalize_input(
            torch.atleast_2d(deepcopy(best_observed_normalized))
        )[0])
        if verbose:
            table = [
                ["Location", "Parameters", "Posterior", "Log-Likelihood"],
                [
                    "MAP",
                    MAP,
                    basq.posterior(MAP_normalized.unsqueeze(0)).cpu().detach(),
                    self.objective_and_loglikelihood_function(
                        MAP_normalized.unsqueeze(0), sober_batch=False
                    )[1].cpu().detach()
                ],
                [
                    "best observed",
                    best_observed,
                    basq.posterior(
                        best_observed_normalized.unsqueeze(0)
                    ).cpu().detach(),
                    self.objective_and_loglikelihood_function(
                        best_observed_normalized.unsqueeze(0),
                        sober_batch=False
                    )[1].cpu().detach()
                ]
            ]
            print(tabulate(table, headers='firstrow', tablefmt='pretty'))

        if visualizations:
            # Minimal necessary transformation: re-ordering.
            # This would be most succinctly be done by applying the
            # back_diagonalization matrix, but that would also skew
            # the covariances. So just use diag_order instead.
            orig_order_samples = torch.zeros_like(taken_samples)
            for par_index, raw_index in enumerate(self.diag_order):
                orig_order_samples.T[par_index] = taken_samples.T[raw_index]
            df = DataFrame(orig_order_samples.numpy())
            if verbose:
                df.describe()
            pairgrid = pairplot(df, kind='kde')
            # Rotate the labels for better readability.
            for i in range(self.input_dim):
                plt.setp(
                    pairgrid.axes[i][0].get_yticklabels(),
                    rotation=45,
                    ha='right',
                    rotation_mode='anchor'
                )
                plt.setp(
                    pairgrid.axes[self.input_dim - 1][i].get_xticklabels(),
                    rotation=45,
                    ha='right',
                    rotation_mode='anchor'
                )
            for i in range(self.input_dim):
                for axis in (
                    pairgrid.axes[self.input_dim - 1][i].xaxis,
                    pairgrid.axes[i][0].yaxis,
                ):
                    fmt = "{:.3g}"
                    axis.set_major_formatter(lambda x, _, index=i: fmt.format(
                        self.denormalize_and_reverse_transform_one_variable(
                            x, index
                        )
                    ))
            if self.normalized_true_optimum is not None:
                for i in range(len(self.true_optimum)):
                    pairgrid.axes[i][i].scatter(
                        self.normalized_true_optimum[i],
                        0, s=100, marker='*', color='tab:orange'
                    )
            pairgrid.tight_layout()
            plt.show()

        return (
            taken_samples
            if return_raw_samples else
            self.reverse_transform(self.denormalize_input(taken_samples)),
            MAP,
            best_observed,
            log_expected_marginal_likelihood,
            log_approx_variance_marginal_likelihood,
        )

    def run_SOBER_adaptively(
        self,
        stopping_criterion_variance=0.1,
        adaptive_batchsize_tolerance=0.1,
        sober_iterations_per_convergence_check=1,
        maximum_number_of_batches=10,
        **kwargs
    ):
        """
        Performs SOBER with optimal adaptive heuristics.

        :param stopping_criterion_variance:
            The integral variance of the estimating kernel SOBER will
            try to reach. For details, see Equation (3b) in
            https://doi.org/10.48550/arXiv.2404.12219
        :param adaptive_batchsize_tolerance:
            The relative quadrature tolerance. Weights will be optimally
            chosen as usual, but with the added freedom of violating
            the quadrature as given by the Nyström approximation.
            Will probably lead to setting more weights to zero,
            eliminating them from the batch, hence reducing batchsize.
        :param sober_iterations_per_convergence_check:
            By default, "the" SOBER algorithm performs BASQ after each
            batch to calculate the stopping criterion. You may set the
            BASQ checks to only happen every so often.
        :param maximum_number_of_batches:
            This is the number of batches produced if the stopping
            criterion does not get met beforehand.
        :param **kwargs:
            As additional (and required) keyword arguments, pass the
            values to call ``run_SOBER`` and ``run_BASQ`` with.
            The exception is `sober_iterations`; use
            `maximum_number_of_batches` instead. See the ``run_SOBER``
            and ``run_BASQ`` docstrings for details.
        """
        if kwargs.get['sober_iterations']:
            maximum_number_of_batches = kwargs.get['sober_iterations']
        kwargs['sober_iterations'] = 1
        for n_iter in range(maximum_number_of_batches):
            self.run_SOBER(**kwargs)
            if not n_iter % sober_iterations_per_convergence_check:
                _, _, _, _, log_variance = self.run_BASQ(**kwargs)
                if exp(log_variance) < stopping_criterion_variance:
                    break
