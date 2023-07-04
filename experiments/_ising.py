import os
import torch
import numpy as np
import itertools
from SOBER._prior import BinaryPrior
from SOBER._utils import TensorManager


ISING_GRID_H = 4
ISING_GRID_W = 4
ISING_N_EDGES = 24

def sample_init_points(n_vertices, n_points, random_seed=None):
    """
    :param n_vertices: 1D array
    :param n_points:
    :param random_seed:
    :return:
    """
    if random_seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(random_seed)
    init_points = torch.empty(0).long()
    for _ in range(n_points):
        init_points = torch.cat([init_points, torch.cat([torch.randint(0, int(elm), (1, 1)) for elm in n_vertices], dim=1)], dim=0)
    if random_seed is not None:
        torch.set_rng_state(rng_state)
    return init_points

def generate_ising_interaction(grid_h, grid_w, random_seed=None):
    if random_seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(random_seed)
    horizontal_interaction = ((torch.randint(0, 2, (grid_h * (grid_w - 1), )) * 2 - 1).float() * (torch.rand(grid_h * (grid_w - 1)) * (5 - 0.05) + 0.05)).view(grid_h, grid_w-1)
    vertical_interaction = ((torch.randint(0, 2, ((grid_h - 1) * grid_w, )) * 2 - 1).float() * (torch.rand((grid_h - 1) * grid_w) * (5 - 0.05) + 0.05)).view(grid_h-1, grid_w)
    if random_seed is not None:
        torch.set_rng_state(rng_state)
    return horizontal_interaction, vertical_interaction

def interaction_sparse2dense(bocs_representation):
    assert bocs_representation.size(0) == bocs_representation.size(1)
    grid_size = int(bocs_representation.size(0) ** 0.5)
    horizontal_interaction = torch.zeros(grid_size, grid_size-1)
    vertical_interaction = torch.zeros(grid_size-1, grid_size)
    for i in range(bocs_representation.size(0)):
        r_i = i // grid_size
        c_i = i % grid_size
        for j in range(i + 1, bocs_representation.size(1)):
            r_j = j // grid_size
            c_j = j % grid_size
            if abs(r_i - r_j) + abs(c_i - c_j) > 1:
                assert bocs_representation[i, j] == 0
            elif abs(r_i - r_j) == 1:
                vertical_interaction[min(r_i, r_j), c_i] = bocs_representation[i, j]
            else:
                horizontal_interaction[r_i, min(c_i, c_j)] = bocs_representation[i, j]
    return horizontal_interaction, vertical_interaction


def interaction_dense2sparse(horizontal_interaction, vertical_interaction):
    grid_size = horizontal_interaction.size(0)
    bocs_representation = torch.zeros(grid_size ** 2, grid_size ** 2)
    for i in range(bocs_representation.size(0)):
        r_i = i // grid_size
        c_i = i % grid_size
        for j in range(i + 1, bocs_representation.size(1)):
            r_j = j // grid_size
            c_j = j % grid_size
            if abs(r_i - r_j) + abs(c_i - c_j) > 1:
                assert bocs_representation[i, j] == 0
            elif abs(r_i - r_j) == 1:
                bocs_representation[i, j] = vertical_interaction[min(r_i, r_j), c_i]
            else:
                bocs_representation[i, j] = horizontal_interaction[r_i, min(c_i, c_j)]
    return bocs_representation + bocs_representation.t()


def spin_covariance(interaction, grid_shape):
    horizontal_interaction, vertical_interaction = interaction
    n_vars = horizontal_interaction.shape[0] * vertical_interaction.shape[1]
    spin_cfgs = np.array(list(itertools.product(*([[-1, 1]] * n_vars))))
    density = np.zeros(spin_cfgs.shape[0])
    for i in range(spin_cfgs.shape[0]):
        spin_cfg = spin_cfgs[i].reshape(grid_shape)
        h_comp = spin_cfg[:, :-1] * horizontal_interaction * spin_cfg[:, 1:] * 2
        v_comp = spin_cfg[:-1] * vertical_interaction * spin_cfg[1:] * 2
        log_interaction_energy = np.sum(h_comp) + np.sum(v_comp)
        density[i] = np.exp(log_interaction_energy)
    interaction_partition = np.sum(density)
    density = density / interaction_partition

    covariance = spin_cfgs.T.dot(spin_cfgs * density.reshape((-1, 1)))
    return covariance, interaction_partition


def partition(interaction, grid_shape):
    horizontal_interaction, vertical_interaction = interaction
    n_vars = horizontal_interaction.shape[0] * vertical_interaction.shape[1]
    spin_cfgs = np.array(list(itertools.product(*([[-1, 1]] * n_vars))))
    interaction_partition = 0
    for i in range(spin_cfgs.shape[0]):
        spin_cfg = spin_cfgs[i].reshape(grid_shape)
        h_comp = spin_cfg[:, :-1] * horizontal_interaction * spin_cfg[:, 1:] * 2
        v_comp = spin_cfg[:-1] * vertical_interaction * spin_cfg[1:] * 2
        log_interaction_energy = np.sum(h_comp) + np.sum(v_comp)
        interaction_partition += np.exp(log_interaction_energy)

    return interaction_partition


def log_partition(interaction, grid_shape):
    horizontal_interaction, vertical_interaction = interaction
    n_vars = horizontal_interaction.shape[0] * vertical_interaction.shape[1]
    spin_cfgs = np.array(list(itertools.product(*([[-1, 1]] * n_vars))))
    log_interaction_energy_list = []
    for i in range(spin_cfgs.shape[0]):
        spin_cfg = spin_cfgs[i].reshape(grid_shape)
        h_comp = spin_cfg[:, :-1] * horizontal_interaction * spin_cfg[:, 1:] * 2
        v_comp = spin_cfg[:-1] * vertical_interaction * spin_cfg[1:] * 2
        log_interaction_energy = np.sum(h_comp) + np.sum(v_comp)
        log_interaction_energy_list.append(log_interaction_energy)

    log_interaction_energy_list = np.array(log_interaction_energy_list)
    max_log_interaction_energy = np.max(log_interaction_energy_list)
    interaction_partition = np.sum(np.exp(log_interaction_energy_list - max_log_interaction_energy))

    return np.log(interaction_partition) + max_log_interaction_energy


def ising_dense(interaction_original, interaction_sparsified, covariance, log_partition_original, log_partition_sparsified):
    diff_horizontal = interaction_original[0] - interaction_sparsified[0]
    diff_vertical = interaction_original[1] - interaction_sparsified[1]

    kld = 0
    n_spin = covariance.shape[0]
    for i in range(n_spin):
        i_h, i_v = int(i / ISING_GRID_H), int(i % ISING_GRID_H)
        for j in range(i, n_spin):
            j_h, j_v = int(j / ISING_GRID_H), int(j % ISING_GRID_H)
            if i_h == j_h and abs(i_v - j_v) == 1:
                kld += diff_horizontal[i_h, min(i_v, j_v)] * covariance[i, j]
            elif abs(i_h - j_h) == 1 and i_v == j_v:
                kld += diff_vertical[min(i_h, j_h), i_v] * covariance[i, j]

    return kld * 2 + log_partition_sparsified - log_partition_original


def _bocs_consistency_mapping(x):
    """
    This is for the comparison with BOCS implementation
    :param x:
    :return:
    """
    horizontal_ind = [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 22, 23]
    vertical_ind = sorted([elm for elm in range(24) if elm not in horizontal_ind])
    return x[horizontal_ind].reshape((ISING_GRID_H, ISING_GRID_W - 1)), x[vertical_ind].reshape((ISING_GRID_H - 1, ISING_GRID_W))


class Ising(object):
    """
    Ising Sparsification Problem with the simplest graph
    """
    def __init__(self, lamda, random_seed_pair=(None, None)):
        self.lamda = lamda
        self.n_vertices = np.array([2] * ISING_N_EDGES)
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat([self.suggested_init, sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0), random_seed_pair[1]).long()], dim=0)
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        self.random_seed_info = 'R'.join([str(random_seed_pair[i]).zfill(4) if random_seed_pair[i] is not None else 'None' for i in range(2)])
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)
        interaction = generate_ising_interaction(ISING_GRID_H, ISING_GRID_W, random_seed_pair[0])
        self.interaction = interaction[0].numpy(), interaction[1].numpy()
        self.covariance, self.partition_original = spin_covariance(self.interaction, (ISING_GRID_H, ISING_GRID_W))

    def evaluate(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == len(self.n_vertices)
        return torch.cat([self._evaluate_single(x[i]) for i in range(x.size(0))], dim=0)

    def _evaluate_single(self, x):
        assert x.dim() == 1
        x_h, x_v = _bocs_consistency_mapping(x.numpy())
        interaction_sparsified = x_h * self.interaction[0], x_v * self.interaction[1]
        log_partition_sparsified = log_partition(interaction_sparsified, (ISING_GRID_H, ISING_GRID_W))
        evaluation = ising_dense(interaction_sparsified=interaction_sparsified, interaction_original=self.interaction,
                                 covariance=self.covariance, log_partition_sparsified=log_partition_sparsified,
                                 log_partition_original=np.log(self.partition_original))
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()
    
def setup_ising():
    """
    Set up the experiments with Ising model sparsification task
    
    Return:
    - prior: class, the function of mixed prior
    - TestFunction: class, the function that returns true function value
    """
    n_dims_binary = ISING_N_EDGES # number of dimensions for binary variables
    n_dims = n_dims_binary # total number of dimensions
    
    tm = TensorManager()
    prior = BinaryPrior(n_dims)
    ising = Ising(0.0001)
    
    def eval_objective(x):
        eval_ = ising.evaluate(x)
        return -1 * eval_.squeeze()

    def TestFunction(X):
        return tm.standardise_tensor(
            torch.tensor(
                [eval_objective(x) for x in X]
            ).squeeze()
        )
    return prior, TestFunction
