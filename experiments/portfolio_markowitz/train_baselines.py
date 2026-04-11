"""Train baseline portfolio models and save checkpoints for warm-start runs.

Run this script before train_warmstart.py. The warm-start script expects to find
the model checkpoints written here under
outputs/portfolio_markowitz/baseline/<run_id>/.
"""

import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from decision_learning.benchmarks.portfolio_markowitz.data import genData
from decision_learning.benchmarks.portfolio_markowitz.oracle import opt_oracle
from decision_learning.modeling.loss import (
    CILOLoss,
    DecisionRegretLoss,
    FYLoss,
    MSELoss,
    PGDCALoss,
    PGAdaptiveLoss,
    PGLoss,
    SPOPlusLoss,
)
from decision_learning.modeling.loss_spec import LossSpec
from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.pipeline import expand_hyperparam_grid, run_loss_experiments
from decision_learning.modeling.smoothing import RandomizedSmoothingWrapper
from decision_learning.utils import handle_solver

import logging

logging.basicConfig(level=logging.INFO)


def load_pred_model(model_path: str, model_class_map: dict | None = None, device: str = "cpu") -> torch.nn.Module:
    """Load a saved prediction model payload produced by this script."""
    payload = torch.load(model_path, map_location=device)
    class_name = payload.get("model_class")
    model_kwargs = payload.get("model_kwargs", {})

    if model_class_map is None:
        model_class_map = {"LinearRegression": LinearRegression}
    if class_name not in model_class_map:
        raise ValueError(f"Unknown model_class '{class_name}'. Provide model_class_map.")

    model = model_class_map[class_name](**model_kwargs)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _format_hparams(hparams: dict) -> str:
    if not hparams:
        return "default"
    parts = []
    for k in sorted(hparams.keys()):
        v = hparams[k]
        v_str = str(v).replace(".", "p").replace("-", "m")
        parts.append(f"{k}{v_str}")
    return "_".join(parts)


def _parse_loss_key(loss_key: str) -> tuple[str, dict]:
    name, _, hparam_str = loss_key.rpartition("_")
    if not hparam_str:
        return name, {}
    try:
        import ast

        hparams = ast.literal_eval(hparam_str)
        if isinstance(hparams, dict):
            return name, hparams
    except Exception:
        pass
    return name, {"raw": hparam_str}


def _repo_root() -> Path:
    # experiments/portfolio_markowitz/train_baselines.py -> repo root
    return Path(__file__).resolve().parents[2]


def _run_id() -> str:
    run_id = os.getenv("SLURM_ARRAY_JOB_ID")
    if not run_id:
        raise ValueError("SLURM_ARRAY_JOB_ID is required for run_id.")
    return run_id


def _portfolio_data_dir() -> Path:
    return _repo_root() / "src" / "decision_learning" / "benchmarks" / "portfolio_markowitz"


def _load_portfolio_support_and_cov() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dir = _portfolio_data_dir()
    Sigma = np.loadtxt(data_dir / "cov_120.csv", delimiter=",", skiprows=1)
    dat_120 = np.loadtxt(data_dir / "dat120_withlags.csv", delimiter=",", skiprows=1)

    dat_Y = -dat_120[:, 1:13]
    dat_X = np.hstack((np.ones((dat_120.shape[0], 1)), dat_120[:, 13:]))
    return dat_Y, dat_X, Sigma


def _make_loss_specs(optmodel, num_data: int) -> list[LossSpec]:
    neurips_pg_h_grid = [num_data ** -0.125, num_data ** -0.25, num_data ** -0.5, num_data ** -1]
    sp_extra_h_grid = neurips_pg_h_grid + [num_data ** -2]
    return [
        LossSpec(name="SPOPlus", factory=SPOPlusLoss, init_kwargs={}, aux={"optmodel": optmodel, "is_minimization": True}),
        LossSpec(name="FY", factory=FYLoss, init_kwargs={}, aux={"optmodel": optmodel, "is_minimization": True}),
        LossSpec(
            name="FY_Smooth",
            factory=RandomizedSmoothingWrapper,
            init_kwargs={
                "base_loss": FYLoss(optmodel=optmodel, is_minimization=True),
                "sigma": 0.1,
                "s": 10,
                "control_variate": True,
            },
        ),
        LossSpec(
            name="DecisionRegret_Smooth",
            factory=RandomizedSmoothingWrapper,
            init_kwargs={
                "base_loss": DecisionRegretLoss(optmodel=optmodel, is_minimization=True),
            },
            hyper_grid=expand_hyperparam_grid({
                "sigma": [0.1],
                "s": [10],
                "control_variate": [True],
            }),
        ),
        LossSpec(name="MSE", factory=MSELoss, init_kwargs={}),
        LossSpec(
            name="DBB",
            factory=PGLoss,
            init_kwargs={"h": 15, "finite_diff_type": "F"},
            aux={"optmodel": optmodel, "is_minimization": True},
        ),
        LossSpec(
            name="PG",
            factory=PGLoss,
            init_kwargs={},
            aux={"optmodel": optmodel, "is_minimization": True},
            hyper_grid=expand_hyperparam_grid({
                "h": neurips_pg_h_grid,
                "finite_diff_type": ["B", "C"],
            }),
        ),
        LossSpec(
            name="PGDCA",
            factory=PGDCALoss,
            init_kwargs={},
            aux={"optmodel": optmodel, "is_minimization": True},
            hyper_grid=expand_hyperparam_grid({
                "h": sp_extra_h_grid,
                "update_every": [10, 25, 100],
            }),
        ),
        LossSpec(
            name="CILO",
            factory=CILOLoss,
            init_kwargs={"optmodel": optmodel, "is_minimization": True},
        ),
        # LossSpec(
        #     name="PGAdaptive",
        #     factory=PGAdaptiveLoss,
        #     init_kwargs={"optmodel": optmodel, "is_minimization": True},
        #     hyper_grid=expand_hyperparam_grid({"h": sp_extra_h_grid}),
        # ),
    ]


def _save_models(
    trained_models: dict,
    models_dir: Path,
    generated_data: dict,
    sim: int,
    num_data: int,
    trial: int,
    run_id: str,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for loss_key, model in trained_models.items():
        loss_name, hparams = _parse_loss_key(loss_key)
        hparam_tag = _format_hparams(hparams)
        fpath = models_dir / f"sim{sim}_n{num_data}_trial{trial}_{loss_name}_{hparam_tag}.pt"
        payload = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "model_kwargs": {
                "input_dim": generated_data["feat"].shape[1],
                "output_dim": generated_data["cost"].shape[1],
            },
            "loss_name": loss_name,
            "hyperparameters": hparams,
            "sim": sim,
            "n": num_data,
            "trial": trial,
            "run_id": run_id,
        }
        torch.save(payload, fpath)


def main():
    if len(sys.argv) < 2:
        raise ValueError("Please provide sim index as sys.argv[1], e.g., python train_baselines.py 0")

    torch.manual_seed(105)
    indices_arr = torch.randperm(100000)
    indices_arr_test = torch.randperm(100000)

    sim = int(sys.argv[1])
    n_arr = [200, 400, 800, 1600]
    trials = 50

    exp_arr = []
    for n in n_arr:
        for t in range(trials):
            exp_arr.append([n, t])

    if sim < 0 or sim >= len(exp_arr):
        raise ValueError(f"sim index out of range: {sim}. Must be in [0, {len(exp_arr) - 1}].")

    num_data, trial = exp_arr[sim]
    epochs = 100
    val_size = 200
    test_size = 2000
    batch_size = 32
    vol_scaling = 0.5
    gamma = 0.1
    save_models = True
    logging.info(f"Running portfolio experiment sim={sim} with n={num_data}, trial={trial}")

    dat_Y, dat_X, Sigma = _load_portfolio_support_and_cov()
    Sigma_inv = np.linalg.inv(Sigma)
    sqrt_det_Sigma = float(np.sqrt(np.linalg.det(Sigma)))

    data_seed = int(indices_arr[sim].item())
    split_seed = int(indices_arr_test[sim].item())
    generated_data = genData(
        num_data=num_data + val_size,
        dat_Y=dat_Y,
        dat_X=dat_X,
        Sigma=Sigma * vol_scaling,
        Sigma_inv=Sigma_inv / vol_scaling,
        sqrt_det_Sigma=sqrt_det_Sigma,
        seed=data_seed,
    )
    generated_data_test = genData(
        num_data=test_size,
        dat_Y=dat_Y,
        dat_X=dat_X,
        Sigma=Sigma * vol_scaling,
        Sigma_inv=Sigma_inv / vol_scaling,
        sqrt_det_Sigma=sqrt_det_Sigma,
        seed=1000,
    )

    portfolio_oracle = partial(opt_oracle, Sigma=Sigma, gamma=gamma)
    optmodel = partial(handle_solver, optmodel=portfolio_oracle, detach_tensor=False, solver_batch_solve=True)

    pred_model = LinearRegression(
        input_dim=generated_data["feat"].shape[1],
        output_dim=generated_data["cost"].shape[1],
    )

    loss_specs = _make_loss_specs(optmodel=optmodel, num_data=num_data)
    train_config = {
        "num_epochs": epochs,
        "lr": 1e-2,
        "dataloader_params": {"batch_size": batch_size, "shuffle": True},
        "scheduler_params": None,
    }

    results_df, trained_models = run_loss_experiments(
        X_train=generated_data["feat"],
        obs_cost_train=generated_data["cost"],
        X_test=generated_data_test["feat"],
        obs_cost_test=generated_data_test["cost"],
        pred_model=pred_model,
        opt_oracle=optmodel,
        train_instance_kwargs={},
        test_instance_kwargs={},
        train_val_split_params={"test_size": val_size, "random_state": split_seed},
        loss_specs=loss_specs,
        train_config=train_config,
        save_models=save_models,
        cond_exp_cost_train=None,
        cond_exp_cost_test=generated_data_test["cond_exp_cost"],
    )

    run_id = _run_id()
    repo_root = _repo_root()
    results_run_dir = repo_root / "outputs" / "portfolio_markowitz" / "baseline" / str(run_id)
    results_run_dir.mkdir(parents=True, exist_ok=True)

    if save_models:
        _save_models(
            trained_models=trained_models,
            models_dir=results_run_dir,
            generated_data=generated_data,
            sim=sim,
            num_data=num_data,
            trial=trial,
            run_id=run_id,
        )

    results_df["sim"] = sim
    results_df["n"] = num_data
    results_df["trial"] = trial
    results_df["data_seed"] = data_seed
    results_df["split_seed"] = split_seed
    results_df["val_metric_cost"] = "observed"
    results_df["test_regret_cost"] = "conditional_expectation"

    results_run_path = results_run_dir / f"sim{sim}_n{num_data}_trial{trial}_results.csv"
    results_df.to_csv(results_run_path, index=False)
    logging.info(f"Wrote results to {results_run_path}")

    group_cols = ["loss_name", "hyperparameters", "sim", "n", "trial"]
    summary_rows = []
    for _, group in results_df.groupby(group_cols, dropna=False):
        group_sorted = group.sort_values("epoch")
        best_idx = group_sorted["val_metric"].idxmin()
        best_row = group_sorted.loc[best_idx]
        last_row = group_sorted.iloc[-1]
        summary_rows.append({
            "loss_name": best_row["loss_name"],
            "hyperparameters": best_row["hyperparameters"],
            "sim": sim,
            "n": num_data,
            "trial": trial,
            "best_val_epoch": int(best_row["epoch"]),
            "test_regret_at_best_val": best_row["test_regret"],
            "test_regret_last_epoch": last_row["test_regret"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_run_path = results_run_dir / f"sim{sim}_n{num_data}_trial{trial}_summary.csv"
    summary_df.to_csv(summary_run_path, index=False)
    logging.info(f"Wrote summary to {summary_run_path}")


if __name__ == "__main__":
    main()
