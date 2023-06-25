import os
import torch
import numpy as np
from collections import OrderedDict
from SOBER._prior import CategoricalPrior
from abc import abstractmethod

PESTCONTROL_N_CHOICE = 5
PESTCONTROL_N_STAGES = 15

class TestFunctionTemplate:
    """
    The abstract class for all benchmark functions acting as objective functions for BO.
    Note that we assume all problems will be minimization problem, so convert maximisation problems as appropriate.
    """

    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = 'categorical'

    def __init__(self, normalize=True, **kwargs):
        self.normalize = normalize
        self.n_vertices = None
        self.config = None
        self.dim = None
        self.continuous_dims = None
        self.categorical_dims = None
        self.int_constrained_dims = None

    def _check_int_constrained_dims(self):
        if self.int_constrained_dims is None:
            return
        assert self.continuous_dims is not None, 'int_constrained_dims must be a subset of the continuous_dims, ' \
                                                 'but continuous_dims is not supplied!'
        int_dims_np = np.asarray(self.int_constrained_dims)
        cont_dims_np = np.asarray(self.continuous_dims)
        assert np.all(np.in1d(int_dims_np, cont_dims_np)), "all continuous dimensions with integer " \
                                                           "constraint must be themselves contained in the " \
                                                           "continuous_dimensions!"

    @abstractmethod
    def compute(self, x, normalize=None):
        raise NotImplementedError()

    def sample_normalize(self, size=None):
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for i in range(size):
            x = np.array([np.random.choice(self.config[_]) for _ in range(self.dim)])
            y.append(self.compute(x, normalize=False, ))
        y = np.array(y)
        return np.mean(y), np.std(y)

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)

def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(x, seed=None):
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    if seed is not None:
        init_pest_frac = np.random.RandomState(seed).beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    else:
        init_pest_frac = np.random.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        if seed is not None:
            spread_rate = np.random.RandomState(seed).beta(spread_alpha, spread_beta, size=(n_simulations,))
        else:
            spread_rate = np.random.beta(spread_alpha, spread_beta, size=(n_simulations,))
        do_control = x[i] > 0
        if do_control:
            if seed is not None:
                control_rate = np.random.RandomState(seed).beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            else:
                control_rate = np.random.beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, control_rate, True)
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                    1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
        else:
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold


class PestControl(TestFunctionTemplate):
    """
	Pest Control Problem.
	"""

    def __init__(self, random_seed=0,
                 normalize=True):
        super(PestControl, self).__init__(normalize)
        self.n_vertices = np.array([PESTCONTROL_N_CHOICE] * PESTCONTROL_N_STAGES)
        self.seed = random_seed
        self.config = self.n_vertices


        self.random_seed_info = str(random_seed).zfill(4)

        self.dim = PESTCONTROL_N_STAGES
        self.categorical_dims = np.arange(self.dim)
        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

    def compute(self, x, normalize=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.int()
        if x.dim() == 1:
            x = x.reshape(1, -1)
        res = torch.tensor([self._compute(x_, normalize) for x_ in x])
        # Add a small ammount of noise to prevent training instabilities
        res += 1e-6 * torch.randn_like(res)
        return res

    def _compute(self, x, normalize=None):
        if normalize is None: normalize = self.normalize
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = _pest_control_score((x.cpu() if x.is_cuda else x).numpy(), seed=self.seed)
        # evaluation = _pest_control_score((x.cpu() if x.is_cuda else x).numpy(), seed=None)
        res = float(evaluation) * x.new_ones((1,)).float()
        if normalize:
            assert self.mean is not None and self.std is not None
            res = (res - self.mean) / self.std
        return res
    
def setup_pest():
    """
    Set up the experiments with Pest Control task
    
    Return:
    - prior: class, the function of mixed prior
    - TestFunction: class, the function that returns true function value
    """
    n_dims_disc = PESTCONTROL_N_STAGES # number of dimensions for categorical variables
    n_discrete = PESTCONTROL_N_CHOICE  # number of categories for categorical variables
    n_dims = n_dims_disc # total number of dimensions
    _min, _max = 0, 4 # the lower and upper bound of categorical varibales
    
    prior = CategoricalPrior(n_dims, _min, _max, n_discrete)
    pest = PestControl(normalize=False)
    
    def eval_objective(x):
        eval_ = pest.compute(x)
        return -1 * eval_.squeeze()

    def TestFunction(X):
        return torch.tensor(
            [eval_objective(x) for x in X]
        ).squeeze()
    
    return prior, TestFunction
