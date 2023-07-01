from itertools import zip_longest
from torch import broadcast_to, erfc, eye, tril, diagonal
from .Phi import Phi

def broadcast_shape(a,b):
    res = reversed(tuple(i if j==1 else (j if i==1 else (i if i==j else -1)) for i,j in zip_longest(reversed(a),reversed(b),fillvalue=1)))
    return list(res)

def PhiDiagonal(z):
    return (erfc(-z*0.70710678118654746171500846685)/2).prod(-1)
    #                  ^ sqrt(2)/2

def multivariate_normal_cdf(value,loc=0.0,covariance_matrix=None,diagonality_tolerance=0.0):
    """Compute orthant probabilities ``P(Z_i < value_i, i = 1,...,d)`` for a multivariate normal random vector Z.
    Closed-form backward differentiation with respect to `value`, `loc` or `covariance_matrix` is supported.

    Parameters
    ----------
    value : torch.Tensor,
        upper integration limits. It can have batch shape.
        The last dimension must be equal to d, the dimension of the
        Gaussian vector.
    loc : torch.Tensor, optional
        Mean of the Gaussian vector. Default is zeros. Can have batch
        shape. Last dimension must be equal to d, the dimension of the
        Gaussian vector. If a float is provided, the value is repeated
        for all the d components.
    covariance_matrix : torch.Tensor, optional
        Covariance matrix of the Gaussian vector.
        Can have batch shape. The two last dimensions must be equal
        to d. Identity matrix by default.
    diagonality_tolerance=0.0 : float, optional
        Avoid expensive numerical integration if the maximum of all
        off-diagonal values is below this tolerance (in absolute value),
        as the covariance is considered diagonal. If there is a batch of
        covariances (e.g. `covariance_matrix` has shape [N,d,d]), then
        the numerical integrations are avoided only if *all* covariances
        are considered diagonal. Diagonality check can be avoided with
        a negative value.
    Returns
    -------
    probability : torch.Tensor
        The probability of the event ``Y < value``. Its shape is the
        the broadcasted batch shape (just a scalar if the batchshape is []).
        Closed form derivative are implemented if `value`  `loc`,
        `covariance_matrix` require a gradient.
    Notes
    -------
    Parameters `value` and `covariance_matrix`, as 
    well as the returned probability tensor are broadcasted to their
    common batch shape. See PyTorch' `broadcasting semantics
    <https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics>`_.
    The integration is performed with Scipy's impementation of A. Genz method [1]_.
    Partial derivative are computed using closed form formula, see e.g. Marmin et al. [2]_, p 13.
    References
    ----------
    .. [1] Alan Genz and Frank Bretz, "Comparison of Methods for the Computation of Multivariate 
       t-Probabilities", Journal of Computational and Graphical Statistics 11, pp. 950-971, 2002. `Source code <http://www.math.wsu.edu/faculty/genz/software/fort77/mvtdstpack.f>`_.
    .. [2] Sébastien Marmin, Clément Chevalier and David Ginsbourger, "Differentiating the multipoint Expected Improvement for optimal batch design", International Workshop on Machine learning, Optimization and big Data, Taormina, Italy, 2015. `PDF <https://hal.archives-ouvertes.fr/hal-01133220v4/document>`_.
    Examples
    --------
    >>> import torch
    >>> from torch.autograd import grad
    >>> from mvnorm import multivariate_normal_cdf as Phi
    >>> n = 4
    >>> x = 1 + torch.randn(n)
    >>> x.requires_grad = True
    >>> # Make a positive semi-definite matrix
    >>> A = torch.randn(n,n)
    >>> C = 1/n*torch.matmul(A,A.t())
    >>> p = Phi(x,covariance_matrix=C)
    >>> p
    tensor(0.3721, grad_fn=<PhiHighDimBackward>)
    >>> grad(p,(x,))[0]
    tensor([0.0085, 0.2510, 0.1272, 0.0332])
    """
    m = loc-value # actually do P(Y-value<0)
    m_shape = m.shape
    d = m_shape[-1]
    if covariance_matrix is None:
        covariance_matrix = eye(d)
        off_diag = -0.0
    else:
        if diagonality_tolerance>=0:
            if d>=2:
                off_diag = tril(covariance_matrix.detach(),diagonal = -1).abs().max()
            else:
                off_diag = -0.0
        else: # diagonality check forbidden by user
            off_diag = diagonality_tolerance + 1
    if off_diag<=diagonality_tolerance: # assumed diagonal
        D = diagonal(covariance_matrix,dim1 = -2, dim2 = -1)
        z = -m/D.sqrt()
        return PhiDiagonal(z)
    cov_shape = covariance_matrix.shape[-2:]
    if len(cov_shape) < 2:
        raise ValueError("covariance_matrix must have at last " \
                         "two dimensions when not diagonal.")
    if cov_shape[-2] != d or cov_shape[-1] != d:
        raise ValueError("Covariance matrix must have the last two " \
                         "dimensions equal to d. Here it's "+str(list(cov_shape[-2:])))
    batch_shape = broadcast_shape(m.shape[:-1],cov_shape[:-2])
    vector_shape = batch_shape + [d]
    matrix_shape = batch_shape + [d,d]
    m_b = broadcast_to(m,vector_shape)
    c_b = broadcast_to(covariance_matrix,matrix_shape)
    return Phi(m_b,c_b)