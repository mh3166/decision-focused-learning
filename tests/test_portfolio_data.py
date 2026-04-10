import numpy as np

from decision_learning.benchmarks.portfolio_markowitz.data import genData


def test_portfolio_gen_data_smoke():
    num_data = 10

    data = genData(num_data=num_data, seed=123)

    assert data["feat"].shape == (num_data, 13)
    assert data["cond_exp_cost"].shape == (num_data, 12)
    assert data["cost"].shape == (num_data, 12)
    assert data["epsilon"].shape == (num_data, 12)
    assert data["support_idx"].shape == (num_data,)

    for key in ["feat", "cond_exp_cost", "cost", "epsilon"]:
        assert np.isfinite(data[key]).all()

    assert np.allclose(data["epsilon"], data["cost"] - data["cond_exp_cost"])


def test_portfolio_gen_data_accepts_custom_support_without_intercept():
    support_y = np.arange(60, dtype=float).reshape(5, 12)
    support_x = np.arange(60, dtype=float).reshape(5, 12) / 10
    Sigma = np.eye(12) * 0.1

    data = genData(
        num_data=3,
        dat_Y=support_y,
        dat_X=support_x,
        Sigma=Sigma,
        seed=123,
    )

    assert data["feat"].shape == (3, 13)
    assert np.allclose(data["feat"][:, 0], 1.0)
    assert data["cond_exp_cost"].shape == (3, 12)
    assert np.isfinite(data["cond_exp_cost"]).all()
