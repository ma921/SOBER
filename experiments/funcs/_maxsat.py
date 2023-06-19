from abc import abstractmethod
import numpy as np
import os
import torch

MAXSAT_DIR_NAME = './dataset'
ISING_GRID_H = 4
ISING_GRID_W = 4
ISING_N_EDGES = 24


class TestFunction:
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


class _MaxSAT(TestFunction):
	def __init__(self, data_filename, random_seed, normalize=False,  **kwargs):
		super(_MaxSAT, self).__init__(normalize, **kwargs)
		f = open(os.path.join(MAXSAT_DIR_NAME, data_filename), 'rt')
		line_str = f.readline()
		while line_str[:2] != 'p ':
			line_str = f.readline()
		self.n_variables = int(line_str.split(' ')[2])
		self.n_clauses = int(line_str.split(' ')[3])
		self.n_vertices = np.array([2] * self.n_variables)
		self.config = self.n_vertices
		clauses = [(float(clause_str.split(' ')[0]), clause_str.split(' ')[1:-1]) for clause_str in f.readlines()]
		f.close()
		weights = np.array([elm[0] for elm in clauses]).astype(np.float32)
		weight_mean = np.mean(weights)
		weight_std = np.std(weights)
		self.weights = (weights - weight_mean) / weight_std
		self.clauses = [([abs(int(elm)) - 1 for elm in clause], [int(elm) > 0 for elm in clause]) for _, clause in clauses]

	def compute(self, x, normalize=None):
		if not isinstance(x, torch.Tensor):
			try:
				x = torch.tensor(x.astype(int))
			except:
				raise Exception('Unable to convert x to a pytorch tensor!')
		return self.evaluate(x)

	def evaluate(self, x,):
		assert x.numel() == self.n_variables
		if x.dim() == 2:
			x = x.squeeze(0)
		x_np = (x.cpu() if x.is_cuda else x).numpy().astype(bool)
		satisfied = np.array([(x_np[clause[0]] == clause[1]).any() for clause in self.clauses])
		return -np.sum(self.weights * satisfied) * x.float().new_ones(1, 1)

class MaxSAT28(_MaxSAT):
	def __init__(self, random_seed=None):
		super().__init__(data_filename='maxcut-johnson8-2-4.clq.wcnf', random_seed=random_seed)
