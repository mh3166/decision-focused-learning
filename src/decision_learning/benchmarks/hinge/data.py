from typing import Dict

import numpy as np


def genData(
    num_data: int,
    m: float = 0.0,
    alpha: float = 1.0,
    seed: int = 135,
) -> Dict[str, np.ndarray]:
    """Generate hinge-style misspecification data.

    Data procedure:
    - X ~ Unif(0, 2), returned as feat with shape (n, 1)
    - f*(x) is a piecewise linear "hinge" with slope 4 then slope -m, kink at x = 0.55
    - Noise epsilon_alpha = sqrt(alpha) * (zeta - 0.5) + sqrt(1 - alpha) * gamma
      where zeta ~ Exponential(scale=0.5) and gamma ~ N(0, 0.5^2)
    - Y = f*(X) + epsilon_alpha

    Returns a dict with keys:
    - feat: (n, 1)
    - cond_exp_cost: (n, 1) noiseless f*(X)
    - cost: (n, 1) noisy observations Y
    - epsilon: (n, 1) noise
    - x: (n,) 1D convenience vector
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")

    if m < -10 or m > 10:
        raise ValueError(f"m is out of a reasonable range [-10, 10], got {m}.")

    rnd = np.random.RandomState(seed)

    x = rnd.uniform(0.0, 2.0, size=(num_data, 1))
    x1d = x[:, 0]

    # piecewise hinge mean function f* (coauthor version)
    f_true = np.empty_like(x1d)
    mask_left = x1d <= 0.55
    f_true[mask_left] = 4.0 * x1d[mask_left] - 2.0
    f_true[~mask_left] = -m * (x1d[~mask_left] - 0.55) + 0.2
    f_true = f_true.reshape(-1, 1)

    zeta = rnd.exponential(scale=0.5, size=(num_data, 1))
    gamma = rnd.normal(loc=0.0, scale=0.5, size=(num_data, 1))

    epsilon = np.sqrt(alpha) * (zeta - 0.5) + np.sqrt(1.0 - alpha) * gamma
    y = f_true + epsilon

    return {
        "feat": x,
        "cond_exp_cost": f_true,
        "cost": y,
        "epsilon": epsilon,
        "x": x1d,
    }
