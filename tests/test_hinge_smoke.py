import numpy as np
import torch

from decision_learning.benchmarks.hinge.data import genData
from decision_learning.benchmarks.hinge.oracle import opt_oracle


def test_hinge_smoke():
    n = 10
    data = genData(num_data=n, m=0.0, alpha=1.0, seed=123)

    for key in ["feat", "cond_exp_cost", "cost", "epsilon"]:
        assert key in data
        assert data[key].shape == (n, 1)
        assert np.isfinite(data[key]).all()

    costs = torch.tensor(data["cost"], dtype=torch.float32)
    sol, obj = opt_oracle(costs)

    assert sol.shape == costs.shape
    assert torch.isfinite(sol).all()
    assert torch.isfinite(obj).all()

    unique_vals = set(sol.unique().tolist())
    assert unique_vals.issubset({-1.0, 1.0})

    expected = torch.where(costs < 0, torch.ones_like(costs), -torch.ones_like(costs))
    assert torch.equal(sol, expected)

    expected_obj = (costs * sol).sum(dim=1, keepdim=True)
    assert torch.allclose(obj, expected_obj)
