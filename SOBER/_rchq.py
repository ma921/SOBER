import time
import torch
from ._utils import SafeTensorOperator
tm = SafeTensorOperator()

def recombination(
    pts_rec,         # random samples for recombination
    pts_nys,         # number of samples used for approximating kernel with Nystrom method
    num_pts,         # number of samples finally returned
    kernel,          # kernel
    device,          # device
    dtype,           # dtype
    init_weights=None,  # initial weights of the sample for recombination
    calc_obj=None,   # a function for calculating an additional objective function
):
    """
    Args:
        - pts_nys: torch.tensor, subsamples for low-rank approximation via Nyström method
        - pts_rec: torch.tensor, subsamples for empirical measure of kernel recomnbination
        - num_pts: int, number of samples finally returned. In BASQ context, this is equivalent to batch size
        - kernel: function of covariance_matrix = function(X, Y). Positive semi-definite Gram matrix (a.k.a. kernel)
        - device: torch.device, cpu or cuda
        - dtype: torch.dtype, torch.float or torch.double
        - init_weights: torch.tensor, weights for importance sampling if pts_rec is not sampled from the prior
        - calc_obj: a function that returns a Tensor of objective values for all the input points

    Returns:
        - x: torch.tensor, the sparcified samples from pts_rec. The number of samples are determined by self.batch_size
        - w: torch.tensor, the positive weights for kernel quadrature as discretised summation.
    """
    return rc_kernel_svd(pts_rec, pts_nys, num_pts, kernel, tm, mu=init_weights, calc_obj=calc_obj)


def ker_svd_sparsify(pt, s, kernel):
    mat = kernel(pt, pt)
    mat = tm.make_cov_psd(mat)
    _U, S, _ = torch.svd_lowrank(mat, q=s)
    U = -1 * _U.T  # Hermitian
    return S, U


def rc_kernel_svd(samp, pt, s, kernel, tm, mu=None, calc_obj=None):
    # Nystrom method
    _, U = ker_svd_sparsify(pt, s - 1, kernel)
    w_star, idx_star = Mod_Tchernychova_Lyons(
        samp, U, pt, kernel, tm, mu=mu, calc_obj=calc_obj
    )
    return idx_star, w_star


def Mod_Tchernychova_Lyons(samp, U_svd, pt_nys, kernel, tm, mu=None, calc_obj=None, DEBUG=False):
    """
    This function is a modified Tcherynychova_Lyons from
    https://github.com/FraCose/Recombination_Random_Algos/blob/master/recombination.py
    """
    N = len(samp)
    n, length = U_svd.shape
    number_of_sets = 2 * (n + 1)

    if mu is None:
        mu = tm.ones(N) / N

    idx_story = tm.arange(N)
    idx_story = idx_story[mu != 0]
    remaining_points = len(idx_story)

    use_obj = False if calc_obj is None else True
    if use_obj:
        obj = -1 * calc_obj(samp)

    while True:
        if remaining_points <= n + 1:
            idx_star = tm.arange(len(mu))[mu > 0]
            w_star = mu[idx_star]
            return w_star, idx_star

        elif n + 1 < remaining_points <= number_of_sets:
            X_mat = U_svd @ kernel(pt_nys, samp[idx_story])
            if use_obj:
                X_mat = torch.cat((X_mat, torch.reshape(
                    obj[idx_story], (1, -1))), 0)
                X_mat_raw = torch.clone(X_mat[:-1])
            
            w_star, idx_star, x_star, _, ERR, _, _ = Tchernychova_Lyons_CAR(
                X_mat.T, torch.clone(mu[idx_story]), tm, DEBUG)

            if use_obj:
                Xp = X_mat_raw[:, idx_star]
                obj_p = obj[idx_star]
                Xp = torch.cat((Xp, tm.ones(1, len(idx_star))), 0)
                _, _, w_null = torch.linalg.svd(Xp)
                w_null = w_null[-1]
                if torch.dot(obj_p, w_null) < 0:
                    w_null = -w_null

                lm = len(w_star)
                plis = w_null > 0
                alpha = tm.zeros(lm)
                alpha[plis] = w_star[plis] / w_null[plis]
                idx_sp = tm.arange(lm)[plis]
                idx_sp = idx_sp[torch.argmin(alpha[plis])]
                w_star = w_star-alpha[idx_sp]*w_null
                w_star[idx_sp] = 0.

                idx_star = idx_star[w_star > 0]
                w_star = w_star[w_star > 0]

            idx_story = idx_story[idx_star]
            mu[:] = 0.
            mu[idx_story] = w_star
            idx_star = idx_story
            w_star = mu[mu > 0]

            return w_star, idx_star

        number_of_el = int(remaining_points / number_of_sets)

        idx = idx_story[:number_of_el *
                        number_of_sets].reshape(number_of_el, -1)
        
        # compute X_for_nys and X_for_obj
        N_approx = number_of_sets * number_of_el
        _idx_tmp = idx_story[:N_approx].reshape(number_of_el, number_of_sets)
        K = kernel(pt_nys, samp[_idx_tmp]) * mu[_idx_tmp].unsqueeze(1)
        X_for_nys = tm.zeros(length, number_of_sets)
        X_for_nys += K.sum(axis=0)

        N_rest = len(idx_story) - N_approx
        if N_rest > 0:
            idx_rest = idx_story[N_approx : N_approx + N_rest]
            K = kernel(pt_nys, samp[idx_rest]) * mu[idx_rest].unsqueeze(0)
            K_pad = torch.cat(
                (K, tm.zeros(length, number_of_sets - N_rest)),
                dim=1,
            )
            X_for_nys += K_pad

        if use_obj:
            X_for_obj = tm.zeros(1, number_of_sets)
            X_for_obj += (
                obj[_idx_tmp].unsqueeze(1) * mu[_idx_tmp].unsqueeze(1)
            ).sum(axis=0).reshape(-1,1)
            if N_rest > 0:
                mat_obj = (obj[idx_rest].unsqueeze(0) * mu[idx_rest].unsqueeze(0)).reshape(-1,1)
                mat_obj_pad = torch.cat((mat_obj, tm.zeros(number_of_sets - N_rest, 1)), dim=0)
                X_for_obj += mat_obj_pad        
        
        X_tmp_tr = U_svd @ X_for_nys
        if use_obj:
            X_tmp_tr = torch.cat((X_tmp_tr, X_for_obj), 0)
        X_tmp = X_tmp_tr.T
        tot_weights = torch.sum(mu[idx], 0)
        idx_last_part = idx_story[number_of_el * number_of_sets:]

        if len(idx_last_part):
            X_mat = U_svd @ kernel(pt_nys, samp[idx_last_part])
            if use_obj:
                X_mat = torch.cat((X_mat, torch.reshape(
                    obj[idx_last_part], (1, -1))), 0)
            X_tmp[-1] += torch.multiply(
                X_mat.T,
                mu[idx_last_part].unsqueeze(1)
            ).sum(axis=0)
            tot_weights[-1] += torch.sum(mu[idx_last_part], 0)
        
        X_tmp = torch.divide(X_tmp, tot_weights.unsqueeze(0).T)

        # sparsify for the case use_obj is True
        if use_obj:
            X_tmp_raw = torch.clone(X_tmp[:, :n])
            obj_raw = X_tmp[:, -1:].reshape(-1)
        
        w_star, idx_star, _, _, ERR, _, _ = Tchernychova_Lyons_CAR(
            X_tmp, torch.clone(tot_weights), tm
        )

        if use_obj:
            Xp = X_tmp_raw[idx_star].T
            obj_p = obj_raw[idx_star]
            Xp = torch.cat((Xp, tm.ones(1, len(idx_star))), 0)
            _, _, w_null = torch.linalg.svd(Xp)
            w_null = w_null[-1]
            if torch.dot(obj_p, w_null) < 0:
                w_null = -w_null

            lm = len(w_star)
            plis = w_null > 0
            alpha = tm.zeros(lm)
            alpha[plis] = w_star[plis] / w_null[plis]
            idx_sp = tm.arange(lm)[plis]
            idx_sp = idx_sp[torch.argmin(alpha[plis])]
            w_star = w_star-alpha[idx_sp]*w_null
            w_star[idx_sp] = 0.

            idx_star = idx_star[w_star > 0]
            w_star = w_star[w_star > 0]

        idx_tomaintain = idx[:, idx_star].reshape(-1)
        idx_tocancel = tm.ones(idx.shape[1]).to(torch.bool)
        idx_tocancel[idx_star] = 0
        idx_tocancel = idx[:, idx_tocancel].reshape(-1)

        mu[idx_tocancel] = 0.
        mu_tmp = torch.multiply(mu[idx[:, idx_star]], w_star)
        mu_tmp = torch.divide(mu_tmp, tot_weights[idx_star])
        mu[idx_tomaintain] = mu_tmp.reshape(-1)

        idx_tmp = idx_star == number_of_sets - 1
        idx_tmp = tm.arange(len(idx_tmp))[idx_tmp != 0]
        # if idx_star contains the last barycenter, whose set could have more points
        if len(idx_tmp) > 0:
            mu_tmp = torch.multiply(mu[idx_last_part], w_star[idx_tmp])
            mu_tmp = torch.divide(mu_tmp, tot_weights[idx_star[idx_tmp]])
            mu[idx_last_part] = mu_tmp
            idx_tomaintain = torch.cat([idx_tomaintain, idx_last_part])
        else:
            idx_tocancel = torch.cat([idx_tocancel, idx_last_part])
            mu[idx_last_part] = 0.

        idx_story = torch.clone(idx_tomaintain)
        remaining_points = len(idx_story)


def Tchernychova_Lyons_CAR(X, mu, tm, DEBUG=False):
    """
    This functions reduce X from N points to n+1.
    This is taken from https://github.com/FraCose/Recombination_Random_Algos/blob/master/recombination.py
    """
    X = torch.cat([tm.ones(X.size(0)).unsqueeze(0).T, X], dim=1)
    N, n = X.shape
    U, Sigma, V = torch.linalg.svd(X.T)
    U = torch.cat([U, tm.zeros(n, N - n)], dim=1)
    Sigma = torch.cat([Sigma, tm.zeros(N - n)])
    Phi = V[-(N - n):, :].T
    cancelled = tm.null()

    for _ in range(N - n):
        lm = len(mu)
        plis = Phi[:, 0] > 0
        # Added 7th Aug, 2023.
        if plis.sum() == 0:
            break
        
        alpha = tm.zeros(lm)
        alpha[plis] = mu[plis] / Phi[plis, 0]
        idx = tm.arange(lm)[plis]
        idx = idx[torch.argmin(alpha[plis])]

        if len(cancelled) == 0:
            cancelled = idx.unsqueeze(0)
        else:
            cancelled = torch.cat([cancelled, idx.unsqueeze(0)])
        mu[:] = mu - alpha[idx] * Phi[:, 0]
        mu[idx] = 0.

        if DEBUG and (not torch.allclose(torch.sum(mu), 1.)):
            # print("ERROR")
            print("sum ", torch.sum(mu))

        Phi_tmp = Phi[:, 0]
        Phi = Phi[:, 1:]
        Phi = Phi - torch.matmul(
            Phi[idx].unsqueeze(1),
            Phi_tmp.unsqueeze(1).T,
        ).T / Phi_tmp[idx]
        Phi[idx, :] = 0.

    w_star = mu[mu > 0]
    idx_star = tm.arange(N)[mu > 0]
    return w_star, idx_star, torch.nan, torch.nan, 0., torch.nan, torch.nan
