import os
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import torch

from decision_learning.utils import handle_solver
from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.pipeline import run_loss_experiments, expand_hyperparam_grid
from decision_learning.modeling.loss_spec import LossSpec
from decision_learning.modeling.loss import (
    SPOPlusLoss,
    MSELoss,
    FYLoss,
    CILOLoss,
    PGLoss,
    PGAdaptiveLoss,
)
from decision_learning.benchmarks.shortest_path_grid.data import genData
from decision_learning.benchmarks.shortest_path_grid.oracle import opt_oracle

# logging
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
    name, _, hparam_str = loss_key.partition("_")
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


def main():
    if len(sys.argv) < 2:
        raise ValueError("Please provide sim index as sys.argv[1], e.g., python sp_experiment_slurm.py 0")

    # ----------------------- SETUP -----------------------
    torch.manual_seed(105)
    indices_arr = torch.randperm(100000)
    indices_arr_test = torch.randperm(100000)

    sim = int(sys.argv[1])

    # construct experiment data configuration array (same ordering as NeurIPS driver)
    n_arr = [200, 400, 800, 1600]
    ep_arr = ['unif', 'normal']
    trials = 100

    exp_arr = []
    for n in n_arr:
        for ep in ep_arr:
            for t in range(trials):
                exp_arr.append([n, ep, t])

    if sim < 0 or sim >= len(exp_arr):
        raise ValueError(f"sim index out of range: {sim}. Must be in [0, {len(exp_arr) - 1}].")

    exp = exp_arr[sim]
    num_data, ep_type, trial = exp
    logging.info(f"Running experiment sim={sim} with n={num_data}, epsilon type={ep_type}, trial={trial}")

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

    generated_data_test = genData(
        # TEMP: smaller test set for quick local sanity check
        num_data=500,
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

    # prediction model
    pred_model = LinearRegression(
        input_dim=generated_data['feat'].shape[1],
        output_dim=generated_data['cost'].shape[1],
    )

    # loss specs
    loss_specs = [
        LossSpec(name='SPOPlus', factory=SPOPlusLoss, init_kwargs={}, aux={"optmodel": optmodel, "is_minimization": True}),
        LossSpec(name='FY', factory=FYLoss, init_kwargs={}, aux={"optmodel": optmodel, "is_minimization": True}),
        LossSpec(name='MSE', factory=MSELoss, init_kwargs={}),
        LossSpec(
            name='DBB',
            factory=PGLoss,
            init_kwargs={"h": 15, "finite_diff_type": "F"},
            aux={"optmodel": optmodel, "is_minimization": True},
        ),
        LossSpec(
            name='PG',
            factory=PGLoss,
            init_kwargs={},
            aux={"optmodel": optmodel, "is_minimization": True},
            hyper_grid=expand_hyperparam_grid({
                "h": [num_data ** -.125, num_data ** -.25, num_data ** -.5, num_data ** -1],
                "finite_diff_type": ["B", "C"],
                "scale_by_norm": [False, True],
            }),
        ),
        LossSpec(
            name='CILO',
            factory=CILOLoss,
            init_kwargs={"optmodel": optmodel, "is_minimization": True},
        ),
        LossSpec(
            name='PGAdaptive',
            factory=PGAdaptiveLoss,
            init_kwargs={"optmodel": optmodel, "is_minimization": True},
        ),
    ]

    train_config = {
        # TEMP: reduce epochs for quick local sanity check
        'num_epochs': 20,
        'dataloader_params': {'batch_size': 32, 'shuffle': True},
    }

    results_df, trained_models = run_loss_experiments(
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
        save_models=True,
        cond_exp_cost_train=generated_data['cond_exp_cost'],
        cond_exp_cost_test=generated_data_test['cond_exp_cost'],
    )

    # ----------------------- SAVE MODELS -----------------------
    run_id = os.getenv("SLURM_JOB_ID", time.strftime("%Y%m%d_%H%M%S"))
    models_dir = os.path.join("outputs", "shortest_path_grid", "models", str(run_id))
    os.makedirs(models_dir, exist_ok=True)

    for loss_key, model in trained_models.items():
        loss_name, hparams = _parse_loss_key(loss_key)
        hparam_tag = _format_hparams(hparams)
        fname = f"sim{sim}_n{num_data}_ep{ep_type}_trial{trial}_{loss_name}_{hparam_tag}.pt"
        fpath = os.path.join(models_dir, fname)

        payload = {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "model_kwargs": {
                "input_dim": generated_data['feat'].shape[1],
                "output_dim": generated_data['cost'].shape[1],
            },
            "loss_name": loss_name,
            "hyperparameters": hparams,
            "sim": sim,
            "n": num_data,
            "ep_type": ep_type,
            "trial": trial,
            "run_id": run_id,
        }
        torch.save(payload, fpath)

    # ----------------------- SAVE RESULTS -----------------------
    results_df['sim'] = sim
    results_df['n'] = num_data
    results_df['ep_type'] = ep_type
    results_df['trial'] = trial

    results_run_dir = os.path.join("outputs", "shortest_path_grid", "results", str(run_id))
    os.makedirs(results_run_dir, exist_ok=True)
    results_run_path = os.path.join(results_run_dir, f"sp_experiment_{sim}.csv")
    results_df.to_csv(results_run_path, index=False)
    logging.info(f"Wrote results to {results_run_path}")

    # Summary: best-val epoch test regret and last-epoch test regret per loss/hparams
    group_cols = ['loss_name', 'hyperparameters', 'sim', 'n', 'ep_type', 'trial']
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
            'best_val_epoch': int(best_row['epoch']),
            'test_regret_at_best_val': best_row['test_regret'],
            'test_regret_last_epoch': last_row['test_regret'],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_run_dir = os.path.join("outputs", "shortest_path_grid", "summary", str(run_id))
    os.makedirs(summary_run_dir, exist_ok=True)
    summary_run_path = os.path.join(summary_run_dir, f"sp_experiment_{sim}.csv")
    summary_df.to_csv(summary_run_path, index=False)
    logging.info(f"Wrote summary to {summary_run_path}")


if __name__ == "__main__":
    main()
