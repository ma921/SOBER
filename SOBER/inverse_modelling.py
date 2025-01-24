from botorch.fit import fit_gpytorch_mll
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from math import exp
from ._sober_wrapper import SoberWrapper
from scipy.stats import chi2
from ._sober import Sober
import torch
import warnings


class InverseModel(SoberWrapper):

    def __init__(
        self,
        model,
        model_initial_samples=0,
        mean=None,
        covariance=None,
        bounds=None,
        prior='Uniform',
        transforms=None,
        seed=None,
        disable_numpy_mode=False,
        parallelization=True,
        visualizations=False,
        **kwargs
    ):
        """
        :param model:
            A method that takes an array of numbers and returns an array
            of numbers. These are intended to be parameters and model
            evaluation, respectively. This method will provide the
            training data throughout the inverse model training.
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
            Note that this refers to the input of `model` and hence
            defines the target area where the inverse model will be
            trained. This will be internally modified with `transforms`.
        :param prior:
            Defaults to the uniform prior within `bounds`. May be set to
            'Gaussian' using `mean` and `covariance` or
            'Truncated_gaussian' using all three.  Note that this refers
            to the input of `model` and hence defines the target area
            where the inverse model will be trained.
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
        :param **kwargs:
            Additional keyword arguments will be passed to the `model`.
        """
        super().__init__(
            model,
            None,  # data
            model_initial_samples,
            mean,
            covariance,
            bounds,
            prior,
            False,  # maximize
            False,  # use_bolfi
            None,  # weights
            None,  # custom_objective_and_loglikelihood
            transforms,
            seed,
            disable_numpy_mode,
            parallelization,
            visualizations,
            None,  # true_optimum
            standalone=False,
            **kwargs
        )

        self.observations_all = None
        self.observations_all_mean = None
        self.observations_all_std = None
        self.inverse_model = None
        self.update_training_data(initialization=True)
        self.results = []
        self.total_sober_iterations = 0
        self.total_model_samples = []

    def process_evaluations(self, evaluations, sober_batch):
        if not sober_batch:
            return
        if self.observations_all is None:
            self.observations_all = evaluations
        else:
            # De-normalize observations_all before mixing it with the
            # new observations.
            self.observations_all = self.observations_all_mean + (
                self.observations_all_std * self.observations_all
            )
            self.observations_all = torch.cat(
                (self.observations_all, evaluations), dim=0
            )
        # Normalize observations_all and store the normalization
        # parameters. For adding more observations later,
        # de-normalization is needed.
        self.observations_all_mean = self.observations_all.mean(dim=0)
        self.observations_all_std = self.observations_all.std(dim=0)
        self.observations_all = (
            self.observations_all - self.observations_all_mean
        ) / self.observations_all_std
        if self.inverse_model is None:
            self.set_inverse_model(self.X_all, self.observations_all)
        self.optimize_inverse_model()

    def default_objective_function(self, observations):
        """
        Uses the inverse model uncertainty as objective.

        :param observations:
            A ``self.model`` evaluation.
        :return:
            The negative of the logarithm of the inv. model unc..
        """
        return -self(observations).variance.log().sum(axis=1).to(
            device=self.tm.device, dtype=self.tm.dtype
        )

    def set_inverse_model(self, x, observations):
        """
        Sets up a simple RBF or advanced BOLFI surrogate.

        :param x:
            The independent variables.
        :param observations:
            The dependent variables. In this case, model evaluations.
        """
        inverse_model = KroneckerMultiTaskGP(observations, x)
        if self.tm.is_cuda():
            self.inverse_model = inverse_model.cuda()
        else:
            self.inverse_model = inverse_model

    def optimize_inverse_model(self):
        """Trains the inverse model on the new observations."""
        self.inverse_model.train()
        self.inverse_model.likelihood.train()
        self.inverse_model.set_train_data(
            self.observations_all, self.X_all, strict=False
        )
        mll = ExactMarginalLogLikelihood(
            self.inverse_model.likelihood, self.inverse_model
        )
        fit_gpytorch_mll(mll)
        self.inverse_model.eval()
        self.inverse_model.likelihood.eval()

    def update_training_data(self, initialization=False):
        warnings.simplefilter("ignore")
        self.Y_all, self.LL_all = self.objective_and_loglikelihood_function(
            self.X_all, sober_batch=initialization
        )
        # Normalize Y_all and store the normalization parameters.
        # For applying Y_batch later, de-normalization is needed.
        self.Y_all_mean = self.Y_all.mean()
        self.Y_all_std = self.Y_all.std()
        self.Y_all = (self.Y_all - self.Y_all_mean) / self.Y_all_std
        self.weights = 1.0
        self.set_rbf_model(self.X_all, self.Y_all, use_bolfi=self.use_bolfi)
        self.sober = Sober(self.prior, self.surrogate_model)

    def optimize_inverse_model_with_SOBER(
        self,
        stopping_criterion_variance=0.1,
        adaptive_batchsize_tolerance=0.1,
        sober_iterations_per_convergence_check=1,
        sober_iterations_per_training_data_updates=1,
        maximum_number_of_batches=10,
        **kwargs
    ):
        """
        Utilizes SOBER to generate optimal training data.

        :param stopping_criterion_variance:
            The integral variance of the estimating kernel SOBER will
            try to reach. For details, see Equation (3b) in
            https://doi.org/10.48550/arXiv.2404.12219
        :param adaptive_batchsize_tolerance;
            The relative quadrature tolerance. Weights will be optimally
            chosen as usual, but with the added freedom of violating
            the quadrature as given by the Nyström approximation.
            Will probably lead to setting more weights to zero,
            elminating them from the batch and hence reducing batchsize.
        :param sober_iterations_per_convergence_check:
            By default, "the" SOBER algorithm performs BASQ after each
            batch to calculate the stopping criterion. You may set the
            BASQ checks to only happen every so often.
        :param sober_iterations_per_training_data_updates:
            By default, the training data will be updated every SOBER
            batch. This resets SOBER each time, so if you wish to
            explicitly generate more training data than one batch at a
            time, set this to a higher number.
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
        if kwargs.get('sober_iterations'):
            maximum_number_of_batches = kwargs['sober_iterations']
        kwargs['sober_iterations'] = 1
        for n_iter in range(maximum_number_of_batches):
            self.run_SOBER(**kwargs)
            if not n_iter % sober_iterations_per_convergence_check:
                _, _, _, _, log_variance = self.run_BASQ(**kwargs)
                if exp(log_variance) < stopping_criterion_variance:
                    break
            if not n_iter % sober_iterations_per_training_data_updates:
                self.update_training_data()

    def evaluate(
        self,
        observations,
        confidence=0.95,
        one_dimensional_confidence=False,
        normalized_space=False
    ):
        """
        Predicts parameter values for the given observations.

        :param observations:
            A batch x output-shaped torch.Tensor.
        :param confidence:
            The confidence value used for the confidence intervals.
            Defaults to 95 %, which is nearly two standard deviations in
            one dimension, and more in higher dimensions.
        :param one_dimensional_confidence:
            If set to True, the confidence will be applied to each
            dimension separately. So 0.95 confidence will always give
            1.95 standard deviations.
        :param normalized_space:
            Set to True if you want the raw results from normalized
            space. Default is results in the model parameter space.
        :returns:
            A 3-tuple with mean, covariance, and errorbars. In the case
            of normalized_space set to False, the variance will still be
            in the normalized space, as it is not trivial to transform.
        """
        if one_dimensional_confidence:
            deviations = chi2(1).ppf(confidence)**0.5
        else:
            deviations = chi2(self.input_dim).ppf(confidence)**0.5
        if not isinstance(observations, torch.Tensor):
            observations = torch.Tensor(observations)
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        raw_prediction = self(observations)
        mean = raw_prediction.mean
        covariance = raw_prediction.covariance_matrix
        standard_deviation = raw_prediction.variance**0.5
        lower_bounds = mean - deviations * standard_deviation
        upper_bounds = mean + deviations * standard_deviation
        if not normalized_space:
            mean = self.reverse_transform(self.denormalize_input(mean))
            lower_bounds = self.reverse_transform(self.denormalize_input(
                lower_bounds
            ))
            upper_bounds = self.reverse_transform(self.denormalize_input(
                upper_bounds
            ))
        return (mean, covariance, (lower_bounds, upper_bounds))

    def sample(self, observations, sample_size, normalized_space=False):
        """
        Samples the prediction for the given observations.

        :param observations:
            A batch x output-shaped torch.Tensor.
        :param sample_size:
            The number of samples to be generated.
        :param normalized_space:
            Set to True if you want the raw results from normalized
            space. Default is results in the model parameter space.
        :returns:
            A torch.Tensor of the samples.
        """
        if not isinstance(observations, torch.Tensor):
            observations = torch.Tensor(observations)
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        raw_prediction = self(observations)
        samples = raw_prediction.sample(torch.Size([sample_size]))
        if not normalized_space:
            num_observations = samples.shape[1]
            parameter_samples = self.reverse_transform(self.denormalize_input(
                samples.reshape(
                    [sample_size * num_observations, self.input_dim]
                )
            ))
            samples = parameter_samples.reshape(
                [sample_size, num_observations, self.input_dim]
            )
        return samples

    def __call__(self, observations):
        """
        Evaluates observations on the inverse model.

        :param observations:
            A batch x output-shaped torch.Tensor.
        :returns:
            The prediction of the inverse model in normalized space.
        """
        # Normalize the observations, as the inverse model was trained
        # in normalized space.
        observations = (
            observations - self.observations_all_mean
        ) / self.observations_all_std
        with torch.no_grad():
            return self.inverse_model.likelihood(
                self.inverse_model(observations)
            )
