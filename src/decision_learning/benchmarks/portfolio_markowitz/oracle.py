"""Gurobi oracle for the Markowitz portfolio benchmark.

This follows the optimization model from neurips_portfolio_experiment.py while
using the repo's oracle contract: opt_oracle(costs, **kwargs) -> (sol, obj).
"""

from typing import Optional, Tuple

import numpy as np
import torch
import logging

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised only when dependency is absent
    gp = None
    GRB = None


logger = logging.getLogger(__name__)

def _as_numpy(costs) -> Tuple[np.ndarray, bool]:
    if isinstance(costs, torch.Tensor):
        costs_np = costs.detach().cpu().numpy()
        return costs_np, True
    return np.asarray(costs), False


def _to_input_type(array: np.ndarray, costs, was_torch: bool):
    if was_torch:
        return torch.as_tensor(array, dtype=costs.dtype, device=costs.device)
    return array


def _require_gurobi() -> None:
    if gp is None:
        raise ImportError(
            "gurobipy is required for the portfolio Markowitz oracle. "
            "Install gurobipy and configure a valid Gurobi license."
        )


def _solve_single(cost: np.ndarray, Sigma: np.ndarray, gamma: float) -> Tuple[np.ndarray, float]:
    _require_gurobi()

    num_assets = cost.shape[0]
    model = gp.Model("portfolio")
    model.Params.OutputFlag = 0
    x = model.addMVar(num_assets, name="x", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)

    model.ModelSense = GRB.MINIMIZE
    model.setObjective(cost @ x)
    model.addConstr(x @ Sigma @ x <= gamma, name="risk")
    model.addConstr(x.sum() <= 1.0, name="budget")
    model.optimize()

    if model.Status == GRB.SUBOPTIMAL and model.SolCount > 0:
        logger.warning(
            "Portfolio oracle received Gurobi status=13 (SUBOPTIMAL). "
            "Accepting incumbent solution and continuing."
        )
    elif model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed to solve portfolio problem; status={model.Status}.")

    sol = np.asarray(x.X, dtype=float)
    obj = float(cost @ sol)
    return sol, obj


def opt_oracle(
    costs,
    Sigma: Optional[np.ndarray] = None,
    gamma: float = 0.1,
):
    """Markowitz portfolio oracle.

    Solves the same optimization problem used in ``neurips_portfolio_experiment.py``:

        minimize    c^T x
        subject to  x^T Sigma x <= gamma
                    sum(x) <= 1
                    0 <= x <= 1

    Args:
        costs: cost vector(s), either shape ``(d,)`` or batched shape ``(B, d)``.
        Sigma: covariance matrix with shape ``(d, d)``. Must be provided.
        gamma: risk budget for the quadratic variance constraint.

    Returns:
        ``(sol, obj)`` with solution shape matching ``costs`` and objective
        shape ``(1,)`` for a single instance or ``(B, 1)`` for a batch. Outputs
        are returned as torch tensors when ``costs`` is a torch tensor, and as
        NumPy arrays otherwise.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be nonnegative, got {gamma}.")

    costs_np, was_torch = _as_numpy(costs)
    if costs_np.ndim == 1:
        costs_2d = costs_np.reshape(1, -1)
        single = True
    elif costs_np.ndim == 2:
        costs_2d = costs_np
        single = False
    else:
        raise ValueError("costs must be 1D (single instance) or 2D (batched).")

    num_assets = costs_2d.shape[1]
    if Sigma is None:
        raise ValueError("Sigma must be provided for the portfolio Markowitz oracle.")
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.shape != (num_assets, num_assets):
        raise ValueError(f"Sigma must have shape {(num_assets, num_assets)}, got {Sigma.shape}.")

    sols = []
    objs = []
    for cost in costs_2d:
        sol, obj = _solve_single(np.asarray(cost, dtype=float), Sigma=Sigma, gamma=gamma)
        sols.append(sol)
        objs.append(obj)

    sol_arr = np.vstack(sols)
    obj_arr = np.asarray(objs, dtype=float).reshape(-1, 1)

    if single:
        sol_arr = sol_arr.reshape(-1)
        obj_arr = obj_arr.reshape(1)

    return (
        _to_input_type(sol_arr, costs, was_torch),
        _to_input_type(obj_arr, costs, was_torch),
    )
