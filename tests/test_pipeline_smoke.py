import random
from functools import partial

import numpy as np
import pandas as pd
import torch

from decision_learning.benchmarks.shortest_path_grid.data import genData
from decision_learning.benchmarks.shortest_path_grid.oracle import opt_oracle
from decision_learning.modeling.loss import PGLoss, MSELoss
from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.pipeline import run_loss_experiments
from decision_learning.modeling.loss_spec import LossSpec
from decision_learning.modeling.smoothing import RandomizedSmoothingWrapper
from decision_learning.utils import handle_solver


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def _run_pipeline(loss_specs_factory, seed: int = 123, print_metrics: bool = False):
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
        optmodel=opt_oracle,
        detach_tensor=False,
        solver_batch_solve=True,
    )

    # solver expects a per-sample grid size; use the grid dimension for all samples.
    train_instance_kwargs = {'size': np.zeros(n_train) + grid[0]}
    test_instance_kwargs = {'size': np.zeros(n_test) + grid[0]}

    pred_model = LinearRegression(
        input_dim=train_data['feat'].shape[1],
        output_dim=train_data['cost'].shape[1],
    )

    loss_specs = loss_specs_factory(optmodel)
    metrics, trained_models = run_loss_experiments(
        X_train=train_data['feat'],
        obs_cost_train=train_data['cost'],
        cond_exp_cost_train=train_data['cond_exp_cost'],
        X_test=test_data['feat'],
        obs_cost_test=test_data['cond_exp_cost'],
        cond_exp_cost_test=test_data['cond_exp_cost'],
        pred_model=pred_model,
        opt_oracle=optmodel,
        train_instance_kwargs=train_instance_kwargs,
        test_instance_kwargs=test_instance_kwargs,
        train_val_split_params={'test_size': 0.25, 'random_state': seed},
        loss_specs=loss_specs,
        train_config={
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


def _assert_metrics(metrics: pd.DataFrame, loss_names) -> None:
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
    for loss_name in loss_names:
        assert loss_name in set(metrics['loss_name'])

    numeric_cols = metrics.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) > 0
    assert np.isfinite(metrics[numeric_cols].to_numpy()).all()


def test_pipeline_smoke_pg_mse():
    def _specs(optmodel):
        return [
            LossSpec(name="PG", factory=PGLoss, init_kwargs={}, aux={"optmodel": optmodel, "is_minimization": True}),
            LossSpec(name="MSE", factory=MSELoss, init_kwargs={}),
        ]
    metrics, trained_models = _run_pipeline(_specs)
    _assert_metrics(metrics, ["PG", "MSE"])
    assert isinstance(trained_models, dict)


def test_pipeline_smoke_user_defined_smoothed_pg():
    loss_name = "PG_SMOOTHED"

    _set_seeds(123)

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
        seed=123,
        plant_edges=False,
    )
    test_data = genData(
        num_data=n_test,
        num_features=num_features,
        grid=grid,
        deg=deg,
        noise_type='unif',
        noise_width=0.1,
        seed=124,
        plant_edges=False,
    )

    optmodel = partial(
        handle_solver,
        optmodel=opt_oracle,
        detach_tensor=False,
        solver_batch_solve=True,
    )

    def _smoothed_pg_factory(optmodel=None, **kwargs):
        base_loss = PGLoss(
            optmodel=optmodel,
            h=0.1,
            finite_diff_type="B",
            reduction="mean",
            is_minimization=True,
        )
        return RandomizedSmoothingWrapper(
            base_loss=base_loss,
            sigma=0.1,
            s=4,
            seed=0,
            antithetic=False,
            control_variate=False,
            reduction="mean",
        )

    train_instance_kwargs = {'size': np.zeros(n_train) + grid[0]}
    test_instance_kwargs = {'size': np.zeros(n_test) + grid[0]}

    pred_model = LinearRegression(
        input_dim=train_data['feat'].shape[1],
        output_dim=train_data['cost'].shape[1],
    )

    metrics, trained_models = run_loss_experiments(
        X_train=train_data['feat'],
        obs_cost_train=train_data['cost'],
        cond_exp_cost_train=train_data['cond_exp_cost'],
        X_test=test_data['feat'],
        obs_cost_test=test_data['cond_exp_cost'],
        cond_exp_cost_test=test_data['cond_exp_cost'],
        pred_model=pred_model,
        opt_oracle=optmodel,
        train_instance_kwargs=train_instance_kwargs,
        test_instance_kwargs=test_instance_kwargs,
        train_val_split_params={'test_size': 0.25, 'random_state': 123},
        loss_specs=[
            LossSpec(
                name=loss_name,
                factory=_smoothed_pg_factory,
                init_kwargs={},
                aux={"optmodel": optmodel},
                extra_batch_data=None,
            )
        ],
        train_config={
            'num_epochs': 1,
            'lr': 1e-2,
            'dataloader_params': {'batch_size': 4, 'shuffle': True},
            'scheduler_params': None,
        },
        save_models=False,
        training_loop_verbose=False,
    )
    _assert_metrics(metrics, [loss_name])
    assert isinstance(trained_models, dict)
