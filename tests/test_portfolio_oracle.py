import numpy as np
import pytest
import torch
import warnings

from decision_learning.benchmarks.portfolio_markowitz import oracle
from decision_learning.benchmarks.portfolio_markowitz.oracle import opt_oracle


def _solve_or_skip(*args, **kwargs):
    try:
        return opt_oracle(*args, **kwargs)
    except Exception as exc:
        message = str(exc).lower()
        gurobi_error = oracle.gp is not None and isinstance(exc, oracle.gp.GurobiError)
        gurobi_unavailable = isinstance(exc, ImportError) or (gurobi_error and "license" in message)
        if gurobi_unavailable:
            warning_msg = f"Gurobi unavailable in this environment; skipping portfolio oracle test: {exc}"
            warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
            pytest.skip(warning_msg)
        raise


def test_portfolio_oracle_requires_sigma():
    with pytest.raises(ValueError, match="Sigma must be provided"):
        opt_oracle(np.array([1.0, 2.0]))


def test_portfolio_oracle_positive_costs_choose_zero_numpy():
    Sigma = np.eye(2)
    costs = np.array([1.0, 2.0])

    sol, obj = _solve_or_skip(costs, Sigma=Sigma, gamma=1.0)

    assert sol.shape == costs.shape
    assert obj.shape == (1,)
    assert np.allclose(sol, np.zeros_like(costs), atol=1e-6)
    assert np.allclose(obj, [0.0], atol=1e-6)


def test_portfolio_oracle_one_asset_risk_bound_numpy_batch():
    Sigma = np.array([[4.0]])
    costs = np.array([[-1.0], [1.0]])

    sol, obj = _solve_or_skip(costs, Sigma=Sigma, gamma=1.0)

    assert sol.shape == costs.shape
    assert obj.shape == (2, 1)
    assert np.allclose(sol[0], [0.5], atol=1e-6)
    assert np.allclose(obj[0], [-0.5], atol=1e-6)
    assert np.allclose(sol[1], [0.0], atol=1e-6)
    assert np.allclose(obj[1], [0.0], atol=1e-6)


def test_portfolio_oracle_preserves_torch_contract():
    Sigma = np.array([[4.0]])
    costs = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)

    sol, obj = _solve_or_skip(costs, Sigma=Sigma, gamma=1.0)

    assert isinstance(sol, torch.Tensor)
    assert isinstance(obj, torch.Tensor)
    assert sol.dtype == costs.dtype
    assert obj.dtype == costs.dtype
    assert sol.shape == costs.shape
    assert obj.shape == (2, 1)
    assert torch.allclose(sol[:, 0], torch.tensor([0.5, 0.0]), atol=1e-6)
    assert torch.allclose(obj[:, 0], torch.tensor([-0.5, 0.0]), atol=1e-6)


def test_portfolio_oracle_enforces_budget_and_risk_constraints():
    Sigma = np.eye(2)
    costs = np.array([[-2.0, -1.0], [-0.5, -3.0]])
    gamma = 1.0

    sol, obj = _solve_or_skip(costs, Sigma=Sigma, gamma=gamma)

    assert sol.shape == costs.shape
    assert np.all(sol >= -1e-6)
    assert np.all(sol <= 1.0 + 1e-6)
    assert np.all(sol.sum(axis=1) <= 1.0 + 1e-6)
    assert np.all(np.einsum("bi,ij,bj->b", sol, Sigma, sol) <= gamma + 1e-6)
    assert np.allclose(obj, np.sum(costs * sol, axis=1, keepdims=True), atol=1e-6)
