import random
from functools import partial

import numpy as np
import pandas as pd
import torch

from decision_learning.data.shortest_path_grid import genData
from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.pipeline import lossfn_experiment_pipeline
from decision_learning.utils import handle_solver


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#TO DO: this should be factored out to a common solver location
# size is the grid dimension (e.g., 5 for a 5x5 grid); 
#sens is the tie tolerance in the backward pass
def _shortest_path_solver(costs, size, sens: float = 1e-4):
    if isinstance(size, np.ndarray):
        size = int(size[0])
    elif isinstance(size, torch.Tensor):
        size = int(size[0].item())

    if type(size) != int:
        size = int(size)

    #the solver isn't actually generic because of hardcoded "9"
    #Todo: Fix this in the refactor.  for now, just throw.
    assert size == 5 

    # Forward Pass
    starting_ind = 0
    starting_ind_c = 0
    samples = costs.shape[0]
    V_arr = torch.zeros(samples, size ** 2)
    for i in range(0, 2 * (size - 1)):
        num_nodes = min(i + 1, 9 - i)
        num_nodes_next = min(i + 2, 9 - i - 1)
        num_arcs = 2 * (max(num_nodes, num_nodes_next) - 1)
        V_1 = V_arr[:, starting_ind:starting_ind + num_nodes]
        layer_costs = costs[:, starting_ind_c:starting_ind_c + num_arcs]
        l_costs = layer_costs[:, 0::2]
        r_costs = layer_costs[:, 1::2]
        next_V_val_l = torch.ones(samples, num_nodes_next) * float('inf')
        next_V_val_r = torch.ones(samples, num_nodes_next) * float('inf')
        if num_nodes_next > num_nodes:
            next_V_val_l[:, :num_nodes_next - 1] = V_1 + l_costs
            next_V_val_r[:, 1:num_nodes_next] = V_1 + r_costs
        else:
            next_V_val_l = V_1[:, :num_nodes_next] + l_costs
            next_V_val_r = V_1[:, 1:num_nodes_next + 1] + r_costs
        next_V_val = torch.minimum(next_V_val_l, next_V_val_r)
        V_arr[:, starting_ind + num_nodes:starting_ind + num_nodes + num_nodes_next] = next_V_val

        starting_ind += num_nodes
        starting_ind_c += num_arcs

    # Backward Pass
    starting_ind = size ** 2
    starting_ind_c = costs.shape[1]
    prev_act = torch.ones(samples, 1)
    sol = torch.zeros(costs.shape)
    for i in range(2 * (size - 1), 0, -1):
        num_nodes = min(i + 1, 9 - i)
        num_nodes_next = min(i, 9 - i + 1)
        V_1 = V_arr[:, starting_ind - num_nodes:starting_ind]
        V_2 = V_arr[:, starting_ind - num_nodes - num_nodes_next:starting_ind - num_nodes]

        num_arcs = 2 * (max(num_nodes, num_nodes_next) - 1)
        layer_costs = costs[:, starting_ind_c - num_arcs: starting_ind_c]

        if num_nodes < num_nodes_next:
            l_cs_res = ((V_2[:, :num_nodes_next - 1] - V_1 + layer_costs[:, ::2]) < sens) * prev_act
            r_cs_res = ((V_2[:, 1:num_nodes_next] - V_1 + layer_costs[:, 1::2]) < sens) * prev_act
            prev_act = torch.zeros(V_2.shape)
            prev_act[:, :num_nodes_next - 1] += l_cs_res
            prev_act[:, 1:num_nodes_next] += r_cs_res
        else:
            l_cs_res = ((V_2 - V_1[:, :num_nodes - 1] + layer_costs[:, ::2]) < sens) * prev_act[:, :num_nodes - 1]
            r_cs_res = ((V_2 - V_1[:, 1:num_nodes] + layer_costs[:, 1::2]) < sens) * prev_act[:, 1:num_nodes]
            prev_act = torch.zeros(V_2.shape)
            prev_act += l_cs_res
            prev_act += r_cs_res
        cs = torch.zeros(layer_costs.shape)
        cs[:, ::2] = l_cs_res
        cs[:, 1::2] = r_cs_res
        sol[:, starting_ind_c - num_arcs: starting_ind_c] = cs

        starting_ind = starting_ind - num_nodes
        starting_ind_c = starting_ind_c - num_arcs
    # Dimension (samples, num edges)
    obj = torch.sum(sol * costs, axis=1)
    # Dimension (samples, 1)
    sol = sol.to(torch.float32)
    obj = obj.reshape(-1, 1).to(torch.float32)
    return sol, obj


def _run_pipeline(loss_name: str, seed: int = 123, print_metrics: bool = False):
    _set_seeds(seed)

    n_train = 8
    n_test = 8
    grid = (5, 5)
    num_features = 3
    deg = 1

    train_data = genData(
        num_data=n_train,
        num_features=num_features,
        grid=grid,
        deg=deg,
        noise_type='unif',
        noise_width=0.1,
        seed=seed,
        plant_edges=False,
    )
    test_data = genData(
        num_data=n_test,
        num_features=num_features,
        grid=grid,
        deg=deg,
        noise_type='unif',
        noise_width=0.1,
        seed=seed + 1,
        plant_edges=False,
    )

    optmodel = partial(
        handle_solver,
        optmodel=_shortest_path_solver,
        detach_tensor=False,
        solver_batch_solve=True,
    )

    # solver expects a per-sample grid size; use the grid dimension for all samples.
    train_solver_kwargs = {'size': np.zeros(n_train) + grid[0]}
    test_solver_kwargs = {'size': np.zeros(n_test) + grid[0]}

    pred_model = LinearRegression(
        input_dim=train_data['feat'].shape[1],
        output_dim=train_data['cost'].shape[1],
    )

    metrics, trained_models = lossfn_experiment_pipeline(
        X_train=train_data['feat'],
        true_cost_train=train_data['cost'],
        X_test=test_data['feat'],
        true_cost_test=test_data['cost_true'],
        predmodel=pred_model,
        optmodel=optmodel,
        train_solver_kwargs=train_solver_kwargs,
        test_solver_kwargs=test_solver_kwargs,
        val_split_params={'test_size': 0.25, 'random_state': seed},
        loss_names=[loss_name],
        training_configs={
            'num_epochs': 1,
            'lr': 1e-2,
            'dataloader_params': {'batch_size': 4, 'shuffle': True},
            'scheduler_params': None,
        },
        save_models=False,
        training_loop_verbose=False,
    )
    # Set print_metrics=True to view pipeline metrics during test runs.
    if print_metrics:
        print(metrics)
    return metrics, trained_models


def _assert_metrics(metrics: pd.DataFrame, loss_name: str) -> None:
    assert isinstance(metrics, pd.DataFrame)
    assert not metrics.empty

    required_columns = {
        'epoch',
        'train_loss',
        'val_metric',
        'test_regret',
        'loss_name',
        'hyperparameters',
    }
    assert required_columns.issubset(metrics.columns)
    assert loss_name in set(metrics['loss_name'])

    numeric_cols = metrics.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) > 0
    assert np.isfinite(metrics[numeric_cols].to_numpy()).all()


def test_pipeline_smoke_pg():
    metrics, trained_models = _run_pipeline('PG')
    _assert_metrics(metrics, 'PG')
    assert isinstance(trained_models, dict)


def test_pipeline_smoke_mse():
    metrics, trained_models = _run_pipeline('MSE')
    _assert_metrics(metrics, 'MSE')
    assert isinstance(trained_models, dict)
