import glob
import os
import sys
import time
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch

from decision_learning.utils import handle_solver
from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.pipeline import run_loss_experiments, expand_hyperparam_grid
from decision_learning.modeling.loss_spec import LossSpec
from decision_learning.modeling.loss import (
    PGLoss,
    PGDCALoss,
    PGAdaptiveLoss,
)
from decision_learning.benchmarks.shortest_path_grid.data import genData
from decision_learning.benchmarks.shortest_path_grid.oracle import opt_oracle

# logging
import logging
logging.basicConfig(level=logging.INFO)


def load_pred_model(model_path: str, model_class_map: dict | None = None, device: str = "cpu") -> torch.nn.Module:
    """Load a saved prediction model payload produced by sp_experiment_slurm.py."""
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


def _repo_root() -> Path:
    # experiments/shortest_path_grid_exp/sp_experiment_initialized_slurm.py -> repo root
    return Path(__file__).resolve().parents[2]


def _run_id() -> str:
    run_id = os.getenv("SLURM_ARRAY_JOB_ID")
    if not run_id:
        raise ValueError("SLURM_ARRAY_JOB_ID is required for run_id.")
    return run_id


def find_baseline_checkpoint(sim: int, num_data: int, ep_type: str, trial: int, model0: str) -> str:
    models_root = _repo_root() / "outputs" / "shortest_path_grid" / "baseline"
    fname = f"sim{sim}_n{num_data}_ep{ep_type}_trial{trial}_{model0}_default.pt"
    baseline_run_id = os.getenv("BASELINE_RUN_ID")
    if not baseline_run_id:
        raise ValueError("BASELINE_RUN_ID is required to locate baseline models.")
    pattern = str(models_root / baseline_run_id / fname)
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


def main():
    if len(sys.argv) < 2:
        raise ValueError("Please provide sim index as sys.argv[1], e.g., python sp_experiment_initialized_slurm.py 0")

    # ----------------------- SETUP -----------------------
    torch.manual_seed(105)
    indices_arr = torch.randperm(100000)
    indices_arr_test = torch.randperm(100000)

    sim = int(sys.argv[1])

    # construct experiment data configuration array (same ordering as NeurIPS driver)
    n_arr = [200, 400, 800, 1600]
    ep_arr = ['unif', 'normal']
    trials = 100
    model0_arr = ['MSE', 'SPOPlus', 'FY']

    exp_arr = []
    for n in n_arr:
        for ep in ep_arr:
            for t in range(trials):
                for model0 in model0_arr:
                    exp_arr.append([n, ep, t, model0])

    if sim < 0 or sim >= len(exp_arr):
        raise ValueError(
            f"sim index out of range: {sim}. Must be in [0, {len(exp_arr) - 1}] "
            f"for exp_arr size={len(exp_arr)}."
        )

    exp = exp_arr[sim]
    num_data, ep_type, trial, model0 = exp
    logging.info(
        f"Running initialized experiment sim={sim}/{len(exp_arr) - 1} "
        f"with n={num_data}, epsilon type={ep_type}, trial={trial}, model0={model0}"
    )

    # shortest path example data generation configurations
    grid = (5, 5)
    num_feat = 5
    deg = 6
    e = .3

    planted_good_pwl_params = {
        'slope0': 0,
        'int0': 2,
        'slope1': 0,
        'int1': 2,
    }
    planted_bad_pwl_params = {
        'slope0': 4,
        'int0': 0,
        'slope1': 0,
        'int1': 2.2,
    }
    plant_edge = True

    # optimization model
    optmodel = partial(handle_solver, optmodel=opt_oracle, detach_tensor=False, solver_batch_solve=True)

    # ----------------------- DATA -----------------------
    generated_data = genData(
        num_data=num_data + 200,
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

    # TEMP: override test set size to 200 for faster runs
    temp_test_points = 200
    generated_data_test = genData(
        num_data=temp_test_points,
        num_features=num_feat,
        grid=grid,
        deg=deg,
        noise_type=ep_type,
        noise_width=e,
        seed=indices_arr_test[trial],
        plant_edges=plant_edge,
        planted_good_pwl_params=planted_good_pwl_params,
        planted_bad_pwl_params=planted_bad_pwl_params,
    )

    if 'cond_exp_cost' not in generated_data or 'cond_exp_cost' not in generated_data_test:
        raise ValueError("genData did not return 'cond_exp_cost' as expected.")

    train_instance_kwargs = {'size': np.zeros(len(generated_data['cost'])) + 5}
    test_instance_kwargs = {'size': np.zeros(len(generated_data_test['cost'])) + 5}

    # Loss hyperparameters
    h_values = [num_data ** -.125, num_data ** -.25, num_data ** -.5, num_data ** -1]

    # TEMP: override epochs to 20 for faster runs
    temp_num_epochs = 20
    train_config = {
        'num_epochs': temp_num_epochs,
        'dataloader_params': {'batch_size': 32, 'shuffle': True},
    }

    run_id = _run_id()

    # ----------------------- RUN EXPERIMENTS -----------------------
    all_results = []
    all_summaries = []

    model_path = find_baseline_checkpoint(sim, num_data, ep_type, trial, model0)
    logging.info(f"Loading baseline model0={model0} from {model_path}")
    pred_model = load_pred_model(model_path)

    loss_specs = [
        LossSpec(
            name='PG',
            factory=PGLoss,
            init_kwargs={},
            aux={"optmodel": optmodel, "is_minimization": True},
            hyper_grid=expand_hyperparam_grid({
                "h": h_values,
                "finite_diff_type": ["B", "C"],
                "scale_by_norm": [False, True],
            }),
        ),
        LossSpec(
            name='PGDCA',
            factory=PGDCALoss,
            init_kwargs={},
            aux={"optmodel": optmodel, "is_minimization": True},
            hyper_grid=expand_hyperparam_grid({
                "h": h_values,
                "update_every": [10, 25, 100],
            }),
        ),
        LossSpec(
            name='PGAdaptive',
            factory=PGAdaptiveLoss,
            init_kwargs={"optmodel": optmodel, "is_minimization": True},
            hyper_grid=expand_hyperparam_grid({
                "h": h_values,
            }),
        ),
    ]

    results_df, _ = run_loss_experiments(
        X_train=generated_data['feat'],
        obs_cost_train=generated_data['cost'],
        X_test=generated_data_test['feat'],
        obs_cost_test=generated_data_test['cost'],
        pred_model=pred_model,
        opt_oracle=optmodel,
        train_instance_kwargs=train_instance_kwargs,
        test_instance_kwargs=test_instance_kwargs,
        train_val_split_params={'test_size': 200, 'random_state': 42},
        loss_specs=loss_specs,
        train_config=train_config,
        save_models=False,
        cond_exp_cost_train=generated_data['cond_exp_cost'],
        cond_exp_cost_test=generated_data_test['cond_exp_cost'],
    )

    results_df['sim'] = sim
    results_df['n'] = num_data
    results_df['ep_type'] = ep_type
    results_df['trial'] = trial
    results_df['model0'] = model0

        # Summary: best-val epoch test regret and last-epoch test regret per loss/hparams
    group_cols = ['loss_name', 'hyperparameters', 'sim', 'n', 'ep_type', 'trial', 'model0']
    summary_rows = []
    for _, group in results_df.groupby(group_cols, dropna=False):
        group_sorted = group.sort_values('epoch')
        best_idx = group_sorted['val_metric'].idxmin()
        best_row = group_sorted.loc[best_idx]
        last_row = group_sorted.iloc[-1]
        summary_rows.append({
            'loss_name': best_row['loss_name'],
            'hyperparameters': best_row['hyperparameters'],
            'sim': sim,
            'n': num_data,
            'ep_type': ep_type,
            'trial': trial,
            'model0': model0,
            'best_val_epoch': int(best_row['epoch']),
            'test_regret_at_best_val': best_row['test_regret'],
            'test_regret_last_epoch': last_row['test_regret'],
        })

    summary_df = pd.DataFrame(summary_rows)

    repo_root = _repo_root()
    results_run_dir = os.path.join(repo_root, "outputs", "shortest_path_grid", "initialized", str(run_id))
    os.makedirs(results_run_dir, exist_ok=True)
    results_run_path = os.path.join(
        results_run_dir,
        f"sim{sim}_n{num_data}_ep{ep_type}_trial{trial}_model0{model0}_results.csv",
    )
    results_df.to_csv(results_run_path, index=False)
    logging.info(f"Wrote results to {results_run_path}")

    summary_run_dir = os.path.join(repo_root, "outputs", "shortest_path_grid", "initialized", str(run_id))
    os.makedirs(summary_run_dir, exist_ok=True)
    summary_run_path = os.path.join(
        summary_run_dir,
        f"sim{sim}_n{num_data}_ep{ep_type}_trial{trial}_model0{model0}_summary.csv",
    )
    summary_df.to_csv(summary_run_path, index=False)
    logging.info(f"Wrote summary to {summary_run_path}")

    logging.info(f"Completed initialized run for sim={sim}. Rows: results={len(results_df)}, summary={len(summary_df)}")


if __name__ == "__main__":
    main()
