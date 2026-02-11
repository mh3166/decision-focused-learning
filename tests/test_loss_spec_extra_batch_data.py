import numpy as np
import torch

from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.pipeline import run_loss_experiments
from decision_learning.modeling.loss_spec import LossSpec


def _dummy_optmodel(costs, **instance_kwargs):
    if isinstance(costs, torch.Tensor):
        n, d = costs.shape
        sol = torch.zeros((n, d), dtype=costs.dtype)
        obj = torch.zeros((n, 1), dtype=costs.dtype)
        return sol, obj
    n, d = costs.shape
    sol = np.zeros((n, d), dtype=costs.dtype)
    obj = np.zeros((n, 1), dtype=costs.dtype)
    return sol, obj


class NeedsExtraFieldLoss(torch.nn.Module):
    def forward(
        self,
        pred_cost,
        true_cost=None,
        true_sol=None,
        true_obj=None,
        instance_kwargs=None,
        *,
        extra,
        **kwargs,
    ):
        return pred_cost.mean() + extra.mean() * 0.0


def test_pipeline_extra_batch_data_is_used():
    rng = np.random.default_rng(0)
    n_train = 8
    n_test = 4
    d_in = 3
    d_cost = 5

    X_train = rng.standard_normal((n_train, d_in)).astype(np.float32)
    X_test = rng.standard_normal((n_test, d_in)).astype(np.float32)
    true_cost_train = rng.standard_normal((n_train, d_cost)).astype(np.float32)
    true_cost_test = rng.standard_normal((n_test, d_cost)).astype(np.float32)

    pred_model = LinearRegression(input_dim=d_in, output_dim=d_cost)
    extra = np.ones((n_train,), dtype=np.float32)

    metrics, _ = run_loss_experiments(
        X_train=X_train,
        true_cost_train=true_cost_train,
        X_test=X_test,
        true_cost_test=true_cost_test,
        pred_model=pred_model,
        opt_oracle=_dummy_optmodel,
        train_instance_kwargs={},
        test_instance_kwargs={},
        train_val_split_params={'test_size': 0.25, 'random_state': 0},
        loss_specs=[
            LossSpec(
                name="NeedsExtra",
                factory=NeedsExtraFieldLoss,
                init_kwargs={},
                extra_batch_data={"extra": extra},
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

    assert "loss_name" in metrics.columns
    assert "NeedsExtra" in set(metrics["loss_name"])


def test_pipeline_extra_batch_data_collision():
    rng = np.random.default_rng(1)
    n_train = 8
    n_test = 4
    d_in = 3
    d_cost = 5

    X_train = rng.standard_normal((n_train, d_in)).astype(np.float32)
    X_test = rng.standard_normal((n_test, d_in)).astype(np.float32)
    true_cost_train = rng.standard_normal((n_train, d_cost)).astype(np.float32)
    true_cost_test = rng.standard_normal((n_test, d_cost)).astype(np.float32)

    pred_model = LinearRegression(input_dim=d_in, output_dim=d_cost)

    try:
        run_loss_experiments(
            X_train=X_train,
            true_cost_train=true_cost_train,
            X_test=X_test,
            true_cost_test=true_cost_test,
            pred_model=pred_model,
            opt_oracle=_dummy_optmodel,
            train_instance_kwargs={},
            test_instance_kwargs={},
            train_val_split_params={'test_size': 0.25, 'random_state': 0},
            loss_specs=[
                LossSpec(
                    name="Collision",
                    factory=NeedsExtraFieldLoss,
                    init_kwargs={},
                    extra_batch_data={"X": X_train},
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
    except ValueError as exc:
        assert "collides with existing batch field" in str(exc)
    else:
        raise AssertionError("Expected collision to raise ValueError")
