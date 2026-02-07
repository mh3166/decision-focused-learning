import torch

from decision_learning.benchmarks.shortest_path_grid.data import genData
from decision_learning.benchmarks.shortest_path_grid.oracle import opt_oracle


def test_shortest_path_solver_smoke():
    grid = (5, 5)
    num_data = 4
    num_features = 3

    generated = genData(
        num_data=num_data,
        num_features=num_features,
        grid=grid,
        deg=1,
        noise_type="unif",
        noise_width=0.0,
        seed=123,
        plant_edges=True,
    )

    costs = torch.tensor(generated["cost"], dtype=torch.float32)
    sol, obj = opt_oracle(costs, size=grid[0])

    assert sol.shape[0] == costs.shape[0]
    assert sol.shape[1] == costs.shape[1]
    assert torch.isfinite(sol).all()
    assert (sol >= 0).all()

    if obj is not None:
        obj_t = obj if isinstance(obj, torch.Tensor) else torch.tensor(obj)
        assert torch.isfinite(obj_t).all()
