import logging
from functools import partial

import numpy as np
import torch

from decision_learning.utils import handle_solver
from decision_learning.benchmarks.shortest_path_grid.data import genData
from decision_learning.benchmarks.shortest_path_grid.oracle import opt_oracle


logging.basicConfig(level=logging.INFO)


def main():
    # Compute the full-information oracle objective value under cond_exp_cost.
    # This provides a useful scale/reference point for interpreting regret and
    # other shortest-path experiment results.
    # ----------------------- SETUP -----------------------
    torch.manual_seed(105)
    indices_arr = torch.randperm(100000)

    # experiment config
    num_data = 5000
    ep_arr = ["unif", "normal"]
    trial = 0

    # shortest path example data generation configurations
    grid = (5, 5)
    num_feat = 5
    deg = 6
    e = 0.3

    planted_good_pwl_params = {
        "slope0": 0,
        "int0": 2,
        "slope1": 0,
        "int1": 2,
    }
    planted_bad_pwl_params = {
        "slope0": 4,
        "int0": 0,
        "slope1": 0,
        "int1": 2.2,
    }
    plant_edge = True

    # optimization model
    optmodel = partial(handle_solver, optmodel=opt_oracle, detach_tensor=False, solver_batch_solve=True)

    for ep_type in ep_arr:
        generated_data = genData(
            num_data=num_data,
            num_features=num_feat,
            grid=grid,
            deg=deg,
            noise_type=ep_type,
            noise_width=e,
            seed=indices_arr[trial],
            plant_edges=plant_edge,
            planted_good_pwl_params=planted_good_pwl_params,
            planted_bad_pwl_params=planted_bad_pwl_params,
        )

        if "cond_exp_cost" not in generated_data:
            raise ValueError("genData did not return 'cond_exp_cost' as expected.")

        costs = torch.tensor(generated_data["cond_exp_cost"], dtype=torch.float32)
        size = torch.zeros(len(generated_data["cost"]), dtype=torch.float32) + 5

        _, full_info_obj = optmodel(costs, size=size)

        mean_obj = full_info_obj.mean().item()
        print(f"ep_type={ep_type} mean_full_info_obj={mean_obj}")


if __name__ == "__main__":
    main()
