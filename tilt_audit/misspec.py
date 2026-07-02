"""Score misspecification: spectral-slope tilt on the prior covariance.

P_wrong(k) = P(k) * (max(k,1)/k_pivot)^eps, pivot at the geometric mean of the
resolved band so the contamination is (approximately) a pure slope change —
the baryonic-feedback analog. eps in {0, +-0.1, +-0.3} tonight.
"""
from __future__ import annotations

import numpy as np

from .fields import Basis, grid_to_z


def contaminated_pk(basis: Basis, pk: np.ndarray, eps: float) -> np.ndarray:
    keff = np.maximum(basis.kmag, 1.0)
    k_pivot = np.sqrt(1.0 * (basis.n / 2))
    return pk * (keff / k_pivot) ** eps


def contaminated_pz(basis: Basis, pk: np.ndarray, eps: float) -> np.ndarray:
    return grid_to_z(contaminated_pk(basis, pk, eps), basis)
