import numpy as np
import torch


def opt_oracle(costs, **kwargs):
    """Simple hinge oracle: choose +1 when cost < 0 else -1.

    Accepts costs as (d,) or (B, d) with d == 1 and returns (sol, obj) where
    sol has the same shape as costs and obj is the per-sample objective.
    """
    del kwargs  # unused

    if isinstance(costs, torch.Tensor):
        return _opt_oracle_torch(costs)
    return _opt_oracle_numpy(costs)


def _opt_oracle_torch(costs: torch.Tensor):
    if costs.dim() == 1:
        if costs.numel() != 1:
            raise ValueError("costs must have d == 1 for the hinge oracle.")
        sol = torch.where(costs < 0, torch.ones_like(costs), -torch.ones_like(costs))
        obj = torch.sum(sol * costs).reshape(1)
        return sol, obj
    if costs.dim() == 2:
        if costs.shape[1] != 1:
            raise ValueError("costs must have shape (B, 1) for the hinge oracle.")
        sol = torch.where(costs < 0, torch.ones_like(costs), -torch.ones_like(costs))
        obj = torch.sum(sol * costs, dim=1, keepdim=True)
        return sol, obj
    raise ValueError("costs must be 1D or 2D.")


def _opt_oracle_numpy(costs: np.ndarray):
    costs = np.asarray(costs)
    if costs.ndim == 1:
        if costs.size != 1:
            raise ValueError("costs must have d == 1 for the hinge oracle.")
        sol = np.where(costs < 0, np.ones_like(costs), -np.ones_like(costs)).astype(costs.dtype)
        obj = np.sum(sol * costs).reshape(1)
        return sol, obj
    if costs.ndim == 2:
        if costs.shape[1] != 1:
            raise ValueError("costs must have shape (B, 1) for the hinge oracle.")
        sol = np.where(costs < 0, np.ones_like(costs), -np.ones_like(costs)).astype(costs.dtype)
        obj = np.sum(sol * costs, axis=1, keepdims=True)
        return sol, obj
    raise ValueError("costs must be 1D or 2D.")
