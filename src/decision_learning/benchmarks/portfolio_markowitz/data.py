"""Synthetic Markowitz portfolio data generator.

The generator mirrors the support-mixture setup from
neurips_portfolio_experiment.py, but returns the repo-standard benchmark keys:
feat, cost, cond_exp_cost, and epsilon.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


_DATA_DIR = Path(__file__).resolve().parent
_DEFAULT_RETURNS_PATH = _DATA_DIR / "dat120_withlags.csv"
_DEFAULT_COV_PATH = _DATA_DIR / "cov_120.csv"


def _load_default_support() -> Tuple[np.ndarray, np.ndarray]:
    """Load the 120-point empirical support used by the portfolio experiment."""
    dat_120 = np.loadtxt(_DEFAULT_RETURNS_PATH, delimiter=",", skiprows=1)

    # Match neurips_portfolio_experiment.py: minimize negative returns and use
    # lagged returns as features with an intercept prepended.
    support_y = -dat_120[:, 1:13]
    support_x = np.hstack((np.ones((dat_120.shape[0], 1)), dat_120[:, 13:]))
    return support_y, support_x


def _load_default_cov(vol_scaling: float) -> np.ndarray:
    cov = np.loadtxt(_DEFAULT_COV_PATH, delimiter=",", skiprows=1)
    return cov * vol_scaling


def gen_data_sub(
    num_samples: int,
    support_Y: np.ndarray,
    support_X: np.ndarray,
    Sigma: np.ndarray,
    rnd: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """Sample portfolio observations from the empirical support."""
    num_support_pts = support_Y.shape[0]
    num_assets = support_Y.shape[1]

    support_idx = rnd.randint(num_support_pts, size=num_samples)
    y = support_Y[support_idx, :].copy()
    x = support_X[support_idx, :].copy()
    x[:, 1:] += rnd.multivariate_normal(
        mean=np.zeros(num_assets),
        cov=Sigma,
        size=num_samples,
    )

    return {"feat": x, "cost": y, "support_idx": support_idx}


def mvn_density(
    x: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
    sqrt_det_Sigma: float,
) -> np.ndarray:
    """Evaluate multivariate-normal density up to the shared constant."""
    diff = mu - x
    exponent = -0.5 * np.einsum("ij,jk,ik->i", diff, Sigma_inv, diff)
    return np.exp(exponent) / sqrt_det_Sigma


def fstar(
    X: np.ndarray,
    support_Y: np.ndarray,
    support_X: np.ndarray,
    Sigma_inv: np.ndarray,
    sqrt_det_Sigma: float,
) -> np.ndarray:
    """Compute E[Y | X] under the support-mixture data model."""
    densities = mvn_density(
        x=X[1:],
        mu=support_X[:, 1:],
        Sigma_inv=Sigma_inv,
        sqrt_det_Sigma=sqrt_det_Sigma,
    )
    prob = densities.sum()
    if prob <= 0 or not np.isfinite(prob):
        raise FloatingPointError("Conditional density normalizer is zero or non-finite.")
    return densities @ support_Y / prob


def conditional_expectation(
    X: np.ndarray,
    support_Y: np.ndarray,
    support_X: np.ndarray,
    Sigma_inv: np.ndarray,
    sqrt_det_Sigma: float,
) -> np.ndarray:
    """Vectorized conditional mean E[Y | X] for a batch of features."""
    # Condition only on lagged returns; the intercept column is fixed at one.
    x_lags = X[:, 1:]
    support_lags = support_X[:, 1:]
    diff = x_lags[:, None, :] - support_lags[None, :, :]
    exponent = -0.5 * np.einsum("nkd,df,nkf->nk", diff, Sigma_inv, diff)
    weights = np.exp(exponent) / sqrt_det_Sigma
    normalizer = weights.sum(axis=1, keepdims=True)
    if np.any(normalizer <= 0) or not np.isfinite(normalizer).all():
        raise FloatingPointError("Conditional density normalizer is zero or non-finite.")
    return weights @ support_Y / normalizer


def genData(
    num_data: int,
    dat_Y: Optional[np.ndarray] = None,
    dat_X: Optional[np.ndarray] = None,
    Sigma: Optional[np.ndarray] = None,
    Sigma_inv: Optional[np.ndarray] = None,
    sqrt_det_Sigma: Optional[float] = None,
    vol_scaling: float = 0.5,
    seed: int = 135,
) -> Dict[str, np.ndarray]:
    """Generate synthetic Markowitz portfolio benchmark data.

    This mirrors ``genData_portfolio_2`` from ``neurips_portfolio_experiment.py``
    while returning the dictionary contract used by the other benchmarks.

    Args:
        num_data: number of samples to generate.
        dat_Y: optional support returns/costs with shape ``(m, 12)``. If omitted,
            the bundled 120-month support is loaded and converted to negative
            returns for minimization.
        dat_X: optional support features. Accepts either lag-only shape
            ``(m, 12)`` or intercept-prepended shape ``(m, 13)``.
        Sigma: optional covariance for feature perturbations. If omitted, the
            bundled covariance is scaled by ``vol_scaling``.
        Sigma_inv: optional inverse covariance for conditional expectation.
        sqrt_det_Sigma: optional square root determinant for the density. It
            only affects densities by a common scale factor.
        vol_scaling: scaling applied to the bundled covariance when ``Sigma``
            is omitted.
        seed: random seed for reproducible support sampling and perturbations.

    Returns:
        dict containing:
            - ``feat``: perturbed features, shape ``(num_data, 13)``
            - ``cond_exp_cost``: conditional expected negative returns,
              shape ``(num_data, 12)``
            - ``cost``: sampled negative returns, shape ``(num_data, 12)``
            - ``epsilon``: ``cost - cond_exp_cost``, shape ``(num_data, 12)``
            - ``support_idx``: sampled support row indices, shape ``(num_data,)``
    """
    if num_data <= 0:
        raise ValueError(f"num_data must be positive, got {num_data}.")
    if vol_scaling <= 0:
        raise ValueError(f"vol_scaling must be positive, got {vol_scaling}.")

    if dat_Y is None or dat_X is None:
        default_y, default_x = _load_default_support()
        if dat_Y is None:
            dat_Y = default_y
        if dat_X is None:
            dat_X = default_x

    support_y = np.asarray(dat_Y, dtype=float)
    support_x = np.asarray(dat_X, dtype=float)
    if support_x.ndim != 2 or support_y.ndim != 2:
        raise ValueError("dat_X and dat_Y must both be two-dimensional arrays.")
    if support_x.shape[0] != support_y.shape[0]:
        raise ValueError(
            f"dat_X and dat_Y must have the same number of support rows, got "
            f"{support_x.shape[0]} and {support_y.shape[0]}."
        )
    if support_x.shape[1] == support_y.shape[1]:
        support_x = np.hstack((np.ones((support_x.shape[0], 1)), support_x))
    if support_x.shape[1] != support_y.shape[1] + 1:
        raise ValueError(
            "dat_X must have either the same number of columns as dat_Y "
            "(lag features) or one extra intercept column."
        )

    if Sigma is None:
        Sigma = _load_default_cov(vol_scaling=vol_scaling)
    Sigma = np.asarray(Sigma, dtype=float)
    num_assets = support_y.shape[1]
    if Sigma.shape != (num_assets, num_assets):
        raise ValueError(
            f"Sigma must have shape {(num_assets, num_assets)}, got {Sigma.shape}."
        )

    if Sigma_inv is None:
        Sigma_inv = np.linalg.inv(Sigma)
    else:
        Sigma_inv = np.asarray(Sigma_inv, dtype=float)
    if sqrt_det_Sigma is None:
        sqrt_det_Sigma = float(np.sqrt(np.linalg.det(Sigma)))
    if sqrt_det_Sigma <= 0 or not np.isfinite(sqrt_det_Sigma):
        raise ValueError(
            f"sqrt_det_Sigma must be positive and finite, got {sqrt_det_Sigma}."
        )

    rnd = np.random.RandomState(seed)
    sampled = gen_data_sub(
        num_samples=num_data,
        support_Y=support_y,
        support_X=support_x,
        Sigma=Sigma,
        rnd=rnd,
    )
    cond_exp_cost = conditional_expectation(
        X=sampled["feat"],
        support_Y=support_y,
        support_X=support_x,
        Sigma_inv=Sigma_inv,
        sqrt_det_Sigma=sqrt_det_Sigma,
    )
    cost = sampled["cost"]

    return {
        "feat": sampled["feat"],
        "cond_exp_cost": cond_exp_cost,
        "cost": cost,
        "epsilon": cost - cond_exp_cost,
        "support_idx": sampled["support_idx"],
    }
