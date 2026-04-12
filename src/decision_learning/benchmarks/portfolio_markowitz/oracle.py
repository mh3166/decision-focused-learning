"""Gurobi oracle for the Markowitz portfolio benchmark.

This follows the optimization model from neurips_portfolio_experiment.py while
using the repo's oracle contract: opt_oracle(costs, **kwargs) -> (sol, obj).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


@dataclass
class _PortfolioModelState:
    model: "gp.Model"
    x: "gp.MVar"
    num_assets: int


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


class PortfolioMarkowitzOracle:
    """Stateful callable oracle that reuses Gurobi models across solves.

    The oracle keeps a cached model per ``(Sigma, gamma)`` so it only builds
    the Gurobi model structure once, then overwrites the objective for each
    sample. This reuse applies both within a batched oracle call and across
    repeated oracle calls over the course of training/evaluation.
    """

    def __init__(self):
        self._model_cache: Dict[Tuple[Tuple[int, int], str, bytes, float], _PortfolioModelState] = {}

    @staticmethod
    def _normalize_sigma(Sigma: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(Sigma, dtype=float))

    @staticmethod
    def _cache_key(Sigma: np.ndarray, gamma: float) -> Tuple[Tuple[int, int], str, bytes, float]:
        Sigma_norm = PortfolioMarkowitzOracle._normalize_sigma(Sigma)
        return Sigma_norm.shape, Sigma_norm.dtype.str, Sigma_norm.tobytes(), float(gamma)

    def _get_model_state(self, Sigma: np.ndarray, gamma: float) -> _PortfolioModelState:
        _require_gurobi()

        Sigma_norm = self._normalize_sigma(Sigma)
        key = self._cache_key(Sigma_norm, gamma)
        state = self._model_cache.get(key)
        if state is not None:
            return state

        # Build the portfolio model once for this (Sigma, gamma) pair and reuse
        # it on later calls by updating only the linear objective.
        num_assets = Sigma_norm.shape[0]
        model = gp.Model("portfolio")
        model.Params.OutputFlag = 0
        x = model.addMVar(num_assets, name="x", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)

        model.ModelSense = GRB.MINIMIZE
        model.addConstr(x @ Sigma_norm @ x <= gamma, name="risk")
        model.addConstr(x.sum() <= 1.0, name="budget")

        state = _PortfolioModelState(model=model, x=x, num_assets=num_assets)
        self._model_cache[key] = state
        return state

    def _solve_single(self, cost: np.ndarray, Sigma: np.ndarray, gamma: float) -> Tuple[np.ndarray, float]:
        state = self._get_model_state(Sigma=Sigma, gamma=gamma)
        if cost.shape[0] != state.num_assets:
            raise ValueError(
                f"Cost vector has length {cost.shape[0]}, expected {state.num_assets} for cached portfolio model."
            )

        # Reuse the cached model and swap in the current objective coefficients.
        state.model.setObjective(cost @ state.x)
        state.model.optimize()

        if state.model.Status == GRB.SUBOPTIMAL and state.model.SolCount > 0:
            logger.warning(
                "Portfolio oracle received Gurobi status=13 (SUBOPTIMAL). "
                "Accepting incumbent solution and continuing."
            )
        elif state.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed to solve portfolio problem; status={state.model.Status}.")

        sol = np.asarray(state.x.X, dtype=float)
        obj = float(cost @ sol)
        return sol, obj

    def __call__(
        self,
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
        Sigma = self._normalize_sigma(Sigma)
        if Sigma.shape != (num_assets, num_assets):
            raise ValueError(f"Sigma must have shape {(num_assets, num_assets)}, got {Sigma.shape}.")

        sols = []
        objs = []
        for cost in costs_2d:
            sol, obj = self._solve_single(np.asarray(cost, dtype=float), Sigma=Sigma, gamma=gamma)
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


opt_oracle = PortfolioMarkowitzOracle()
