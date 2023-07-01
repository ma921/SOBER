"""
    - 1. Integrate multivariate normal density (CDFs)
    - 2. Easily obtain partial derivatives of CDFs w.r.t location, mean and covariance (and higher derivatives)
    - 3. Manipulate quantities within a tensor-based framework (e.g. broadcasting is fully supported)

"""

from .multivariate_normal_cdf import multivariate_normal_cdf
from .integration import integration