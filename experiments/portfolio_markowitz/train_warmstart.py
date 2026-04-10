"""Train portfolio PG variants starting from saved baseline checkpoints.

Run train_baselines.py first. This script locates those checkpoints using
BASELINE_RUN_ID, which should be set to the SLURM_ARRAY_JOB_ID from the
baseline batch run. Its own outputs are grouped under the current
SLURM_ARRAY_JOB_ID, or "local" when run outside SLURM.
"""

import glob
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch

from decision_learning.benchmarks.portfolio_markowitz.data import genData
from decision_learning.benchmarks.portfolio_markowitz.oracle import opt_oracle
from decision_learning.modeling.loss import PGAdaptiveLoss, PGDCALoss, PGLoss
from decision_learning.modeling.loss_spec import LossSpec
from decision_learning.modeling.pipeline import expand_hyperparam_grid, run_loss_experiments
from decision_learning.utils import handle_solver
from train_baselines import (
    _load_portfolio_support_and_cov,
    _parse_int_list_env,
    _repo_root,
    load_pred_model,
)

import logging

logging.basicConfig(level=logging.INFO)


def _run_id() -> str:
    return os.getenv("SLURM_ARRAY_JOB_ID") or "local"


def _baseline_run_id() -> str:
    run_id = os.getenv("BASELINE_RUN_ID")
    if not run_id:
        raise ValueError("BASELINE_RUN_ID is required to locate portfolio baseline models.")
    return run_id


def find_baseline_checkpoint(baseline_sim: int, num_data: int, trial: int, model0: str) -> str:
    models_root = _repo_root() / "outputs" / "portfolio_markowitz"
    fname = f"sim{baseline_sim}_n{num_data}_trial{trial}_{model0}_default.pt"
    pattern = str(models_root / _baseline_run_id() / fname)
    matches = sorted(glob.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            "Missing baseline checkpoint. "
            f"Expected path like {pattern}"
        )
    if len(matches) > 1:
        raise FileNotFoundError(
            "Multiple baseline checkpoints found. "
            f"Expected a single match for {pattern}. "
            f"Matches: {matches}"
        )
    return matches[0]


def _make_loss_specs(optmodel, num_data: int) -> list[LossSpec]:
    h_values = [num_data ** -0.125, num_data ** -0.25, num_data ** -0.5, num_data ** -1, num_data ** -2]
    return [
        LossSpec(
            name="PG",
            factory=PGLoss,
            init_kwargs={},
            aux={"optmodel": optmodel, "is_minimization": True},
            hyper_grid=expand_hyperparam_grid({
                "h": h_values,
                "finite_diff_type": ["B", "C"],
            }),
        ),
        LossSpec(
            name="PGDCA",
            factory=PGDCALoss,
            init_kwargs={},
            aux={"optmodel": optmodel, "is_minimization": True},
            hyper_grid=expand_hyperparam_grid({
                "h": h_values,
                "update_every": [10, 25, 100],
            }),
        ),
        # LossSpec(
        #     name="PGAdaptive",
        #     factory=PGAdaptiveLoss,
        #     init_kwargs={"optmodel": optmodel, "is_minimization": True},
        #     hyper_grid=expand_hyperparam_grid({
        #         "h": h_values,
        #     }),
        # ),
    ]


def main():
    if len(sys.argv) < 2:
        raise ValueError("Please provide sim index as sys.argv[1], e.g., python train_warmstart.py 0")

    torch.manual_seed(105)
    indices_arr = torch.randperm(100000)
    indices_arr_test = torch.randperm(100000)

    n_arr = _parse_int_list_env("PORTFOLIO_N_ARR", [200, 400, 800, 1600])
    trials = int(os.getenv("PORTFOLIO_TRIALS", "50"))
    if trials <= 0:
        raise ValueError(f"PORTFOLIO_TRIALS must be positive, got {trials}.")

    model0_arr = ["MSE", "SPOPlus", "FY"]
    baseline_exp_arr = []
    for n in n_arr:
        for t in range(trials):
            baseline_exp_arr.append([n, t])

    exp_arr = []
    for n, t in baseline_exp_arr:
        for model0 in model0_arr:
            exp_arr.append([n, t, model0])

    sim = int(sys.argv[1])
    if sim < 0 or sim >= len(exp_arr):
        raise ValueError(
            f"sim index out of range: {sim}. Must be in [0, {len(exp_arr) - 1}] "
            f"for exp_arr size={len(exp_arr)}."
        )

    num_data, trial, model0 = exp_arr[sim]
    try:
        baseline_sim = baseline_exp_arr.index([num_data, trial])
    except ValueError as exc:
        raise ValueError(f"Could not resolve baseline sim for n={num_data}, trial={trial}.") from exc

    epochs = int(os.getenv("PORTFOLIO_EPOCHS", "100"))
    val_size = int(os.getenv("PORTFOLIO_VAL_SIZE", "200"))
    test_size = int(os.getenv("PORTFOLIO_TEST_SIZE", "2000"))
    batch_size = int(os.getenv("PORTFOLIO_BATCH_SIZE", "32"))
    vol_scaling = float(os.getenv("PORTFOLIO_VOL_SCALING", "0.5"))
    gamma = float(os.getenv("PORTFOLIO_GAMMA", "0.1"))
    if epochs <= 0 or val_size <= 0 or test_size <= 0 or batch_size <= 0:
        raise ValueError("PORTFOLIO_EPOCHS, PORTFOLIO_VAL_SIZE, PORTFOLIO_TEST_SIZE, and PORTFOLIO_BATCH_SIZE must be positive.")

    logging.info(
        f"Running warm-start portfolio experiment sim={sim}/{len(exp_arr) - 1} "
        f"with baseline_sim={baseline_sim}, n={num_data}, trial={trial}, model0={model0}"
    )

    dat_Y, dat_X, Sigma = _load_portfolio_support_and_cov()
    Sigma_inv = np.linalg.inv(Sigma)
    sqrt_det_Sigma = float(np.sqrt(np.linalg.det(Sigma)))

    # Match train_baselines.py path-by-path: the baseline sim determines both
    # data generation and train/validation split seeds.
    data_seed = int(indices_arr[baseline_sim].item())
    split_seed = int(indices_arr_test[baseline_sim].item())
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

    model_path = find_baseline_checkpoint(baseline_sim, num_data, trial, model0)
    logging.info(f"Loading baseline model0={model0} from {model_path}")
    pred_model = load_pred_model(model_path)

    loss_specs = _make_loss_specs(optmodel=optmodel, num_data=num_data)
    train_config = {
        "num_epochs": epochs,
        "lr": 1e-2,
        "dataloader_params": {"batch_size": batch_size, "shuffle": True},
        "scheduler_params": None,
    }

    results_df, _ = run_loss_experiments(
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
        save_models=False,
        cond_exp_cost_train=None,
        cond_exp_cost_test=generated_data_test["cond_exp_cost"],
    )

    results_df["sim"] = sim
    results_df["baseline_sim"] = baseline_sim
    results_df["n"] = num_data
    results_df["trial"] = trial
    results_df["model0"] = model0
    results_df["data_seed"] = data_seed
    results_df["split_seed"] = split_seed
    results_df["val_metric_cost"] = "observed"
    results_df["test_regret_cost"] = "conditional_expectation"

    group_cols = ["loss_name", "hyperparameters", "sim", "baseline_sim", "n", "trial", "model0"]
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
            "baseline_sim": baseline_sim,
            "n": num_data,
            "trial": trial,
            "model0": model0,
            "best_val_epoch": int(best_row["epoch"]),
            "test_regret_at_best_val": best_row["test_regret"],
            "test_regret_last_epoch": last_row["test_regret"],
        })

    summary_df = pd.DataFrame(summary_rows)

    run_id = _run_id()
    results_run_dir = _repo_root() / "outputs" / "portfolio_markowitz" / "warmstart" / str(run_id)
    results_run_dir.mkdir(parents=True, exist_ok=True)

    results_run_path = results_run_dir / f"sim{sim}_n{num_data}_trial{trial}_model0{model0}_results.csv"
    results_df.to_csv(results_run_path, index=False)
    logging.info(f"Wrote results to {results_run_path}")

    summary_run_path = results_run_dir / f"sim{sim}_n{num_data}_trial{trial}_model0{model0}_summary.csv"
    summary_df.to_csv(summary_run_path, index=False)
    logging.info(f"Wrote summary to {summary_run_path}")
    logging.info(f"Completed warm-start run for sim={sim}. Rows: results={len(results_df)}, summary={len(summary_df)}")


if __name__ == "__main__":
    main()
