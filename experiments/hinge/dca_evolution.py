from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from decision_learning.benchmarks.hinge.data import genData
from decision_learning.benchmarks.hinge.oracle import opt_oracle
from decision_learning.modeling.loss import PGDCALoss, PGLoss
from decision_learning.modeling.val_metrics import decision_regret
from decision_learning.utils import handle_solver


# This script illustrates the PGDCA/DCA dynamics on the 1D hinge benchmark.
# It writes two CSV outputs under outputs/hinge:
# - DCA_Landscapes.csv: the fixed PGC loss curve plus PGDCA loss curves
#   recorded immediately after each model0 refresh.
# - DCA_iterates.csv: per-epoch training history including beta, train regret,
#   test regret, train loss, and whether model0 was refreshed at that epoch.

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scale_to_unit_interval(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    xmin = np.min(arr)
    xmax = np.max(arr)
    if xmax == xmin:
        return np.zeros_like(arr, dtype=float)
    return (arr - xmin) / (xmax - xmin)


class BetaModel(nn.Module):
    """One-parameter hinge predictor with fixed intercept."""

    def __init__(self, beta: float = 0.0, intercept: float = -2.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(float(intercept), dtype=torch.float32))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.beta * X + self.intercept


@dataclass
class DatasetBundle:
    X: torch.Tensor
    obs_cost: torch.Tensor
    cond_exp_cost: torch.Tensor
    full_info_obj: torch.Tensor


def _default_args_dict() -> dict[str, object]:
    return {
        "num_data": 200,
        "num_test_data": 5000,
        "m": 0.0,
        "alpha": 1.0,
        "seed": 885309,
        "test_seed": 885310,
        "training_seed": 0,
        "beta_min": 0.0,
        "beta_max": 10.0,
        "num_beta_points": 101,
        "intercept": -2.0,
        "initial_beta": 0.0,
        "init_model": "none",
        "h": 0.2,
        "update_every": 100,
        "num_epochs": 1000,
        "lr": 0.01,
    }


def _format_tag_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace(".", "p").replace("-", "m")


def _output_suffix(args: argparse.Namespace) -> str:
    default_args = _default_args_dict()
    parts = []
    for key in sorted(default_args):
        value = getattr(args, key)
        if value != default_args[key]:
            parts.append(f"{key}{_format_tag_value(value)}")
    return "" if not parts else "__" + "_".join(parts)


def _resolve_output_path(base_name: str, output_arg: Path | None, suffix: str) -> Path:
    if output_arg is not None:
        return output_arg
    return _repo_root() / "outputs" / "hinge" / f"{base_name}{suffix}.csv"


def _make_dataset(*, num_data: int, m: float, alpha: float, seed: int, optmodel: callable) -> DatasetBundle:
    data = genData(num_data=num_data, m=m, alpha=alpha, seed=seed)
    X = torch.tensor(data["feat"], dtype=torch.float32)
    obs_cost = torch.tensor(data["cost"], dtype=torch.float32)
    cond_exp_cost = torch.tensor(data["cond_exp_cost"], dtype=torch.float32)
    _, full_info_obj = optmodel(cond_exp_cost)
    full_info_obj = torch.as_tensor(full_info_obj, dtype=torch.float32)
    return DatasetBundle(
        X=X,
        obs_cost=obs_cost,
        cond_exp_cost=cond_exp_cost,
        full_info_obj=full_info_obj,
    )


def _fit_mse_beta(X: torch.Tensor, obs_cost: torch.Tensor, intercept: float) -> float:
    x = X.reshape(-1)
    y = obs_cost.reshape(-1)
    target = y - intercept
    denom = torch.sum(x * x)
    if abs(float(denom)) < 1e-12:
        return 0.0
    beta = torch.sum(x * target) / denom
    return float(beta.item())


def _calc_regret(model: BetaModel, dataset: DatasetBundle, optmodel: callable) -> float:
    with torch.no_grad():
        pred_cost = model(dataset.X)
    return decision_regret(
        pred_cost=pred_cost,
        cond_exp_cost=dataset.cond_exp_cost,
        full_info_obj=dataset.full_info_obj,
        optmodel=optmodel,
        is_minimization=True,
    )


def _beta_grid(args: argparse.Namespace) -> np.ndarray:
    return np.linspace(args.beta_min, args.beta_max, args.num_beta_points)


def _evaluate_pgc_curve(
    *,
    betas: np.ndarray,
    X: torch.Tensor,
    obs_cost: torch.Tensor,
    intercept: float,
    h: float,
    optmodel: callable,
) -> tuple[list[float], np.ndarray]:
    loss_fn = PGLoss(optmodel=optmodel, h=h, finite_diff_type="C")
    raw_values = []
    for beta in betas:
        pred_cost = torch.tensor(beta, dtype=X.dtype) * X + intercept
        raw_values.append(float(loss_fn(pred_cost=pred_cost, obs_cost=obs_cost).item()))
    return raw_values, _scale_to_unit_interval(raw_values)


def _evaluate_pgdca_curve(
    *,
    betas: np.ndarray,
    X: torch.Tensor,
    obs_cost: torch.Tensor,
    intercept: float,
    h: float,
    model0: nn.Module | None,
    optmodel: callable,
) -> tuple[list[float], np.ndarray]:
    loss_fn = PGDCALoss(
        optmodel=optmodel,
        h=h,
        update_every=0,
        model0=copy.deepcopy(model0) if model0 is not None else None,
    )
    raw_values = []
    for beta in betas:
        pred_model = BetaModel(beta=float(beta), intercept=intercept)
        pred_cost = pred_model(X)
        raw_values.append(
            float(
                loss_fn(
                    pred_cost=pred_cost,
                    obs_cost=obs_cost,
                    X=X,
                    pred_model=pred_model,
                ).item()
            )
        )
    return raw_values, _scale_to_unit_interval(raw_values)


def _landscape_rows(
    *,
    betas: np.ndarray,
    raw_values: list[float],
    scaled_values: np.ndarray,
    loss_name: str,
    record_epoch: int,
    refresh_index: int,
) -> list[dict[str, float | int | str]]:
    rows = []
    for beta, raw_value, scaled_value in zip(betas, raw_values, scaled_values):
        rows.append(
            {
                "record_epoch": record_epoch,
                "refresh_index": refresh_index,
                "loss_name": loss_name,
                "beta": float(beta),
                "loss_value": float(raw_value),
                "loss_value_scaled": float(scaled_value),
            }
        )
    return rows


def run_experiment(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(args.training_seed)
    optmodel = partial(handle_solver, optmodel=opt_oracle, detach_tensor=False, solver_batch_solve=True)

    train_data = _make_dataset(
        num_data=args.num_data,
        m=args.m,
        alpha=args.alpha,
        seed=args.seed,
        optmodel=optmodel,
    )
    test_data = _make_dataset(
        num_data=args.num_test_data,
        m=args.m,
        alpha=args.alpha,
        seed=args.test_seed,
        optmodel=optmodel,
    )

    mse_beta = _fit_mse_beta(train_data.X, train_data.obs_cost, args.intercept)
    initial_beta = mse_beta if args.init_model == "mse" else args.initial_beta
    model = BetaModel(beta=initial_beta, intercept=args.intercept)

    model0 = None
    if args.init_model == "mse":
        model0 = BetaModel(beta=mse_beta, intercept=args.intercept)

    loss_fn = PGDCALoss(
        optmodel=optmodel,
        h=args.h,
        update_every=args.update_every,
        model0=model0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    betas = _beta_grid(args)
    pgc_raw, pgc_scaled = _evaluate_pgc_curve(
        betas=betas,
        X=train_data.X,
        obs_cost=train_data.obs_cost,
        intercept=args.intercept,
        h=args.h,
        optmodel=optmodel,
    )
    landscape_rows = _landscape_rows(
        betas=betas,
        raw_values=pgc_raw,
        scaled_values=pgc_scaled,
        loss_name="pgc_loss",
        # Sentinel values indicate the pre-training reference curve, which is
        # not tied to any model0 refresh during DCA training.
        record_epoch=-1,
        refresh_index=-1,
    )

    iterate_rows: list[dict[str, float | int | bool]] = []
    refresh_count = 0
    for epoch in range(args.num_epochs):
        pred_cost = model(train_data.X)
        loss = loss_fn(
            pred_cost=pred_cost,
            obs_cost=train_data.obs_cost,
            X=train_data.X,
            pred_model=model,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cur_beta = float(model.beta.item())
            train_regret = _calc_regret(model, train_data, optmodel)
            test_regret = _calc_regret(model, test_data, optmodel)

        refreshed = False
        if args.init_model == "none" and epoch == 0:
            refreshed = True
        elif args.update_every > 0 and (epoch + 1) % args.update_every == 0:
            refreshed = True

        iterate_rows.append(
            {
                "epoch": epoch,
                "beta": cur_beta,
                "train_regret": train_regret,
                "test_regret": test_regret,
                "train_loss": float(loss.item()),
                "refreshed_model0": refreshed,
            }
        )

        if refreshed:
            refresh_count += 1
            pgdca_raw, pgdca_scaled = _evaluate_pgdca_curve(
                betas=betas,
                X=train_data.X,
                obs_cost=train_data.obs_cost,
                intercept=args.intercept,
                h=args.h,
                model0=loss_fn.model0,
                optmodel=optmodel,
            )
            landscape_rows.extend(
                _landscape_rows(
                    betas=betas,
                    raw_values=pgdca_raw,
                    scaled_values=pgdca_scaled,
                    loss_name="pgdca_loss",
                    record_epoch=epoch,
                    refresh_index=refresh_count,
                )
            )

    return pd.DataFrame(landscape_rows), pd.DataFrame(iterate_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Illustrate PGDCA evolution on the hinge benchmark.")
    parser.add_argument("--num-data", type=int, default=200)
    parser.add_argument("--num-test-data", type=int, default=5000)
    parser.add_argument("--m", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=885309)
    parser.add_argument("--test-seed", type=int, default=885310)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=10.0)
    parser.add_argument("--num-beta-points", type=int, default=101)
    parser.add_argument("--intercept", type=float, default=-2.0)
    parser.add_argument("--initial-beta", type=float, default=0.0)
    parser.add_argument("--init-model", choices=["none", "mse"], default="none")
    parser.add_argument("--h", type=float, default=0.2)
    parser.add_argument("--update-every", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--landscape-output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--iterate-output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    landscapes_df, iterates_df = run_experiment(args)

    suffix = _output_suffix(args)
    landscape_output = _resolve_output_path("DCA_Landscapes", args.landscape_output, suffix)
    iterate_output = _resolve_output_path("DCA_iterates", args.iterate_output, suffix)

    landscape_output.parent.mkdir(parents=True, exist_ok=True)
    iterate_output.parent.mkdir(parents=True, exist_ok=True)

    landscapes_df.to_csv(landscape_output, index=False)
    iterates_df.to_csv(iterate_output, index=False)

    print(f"Wrote {len(landscapes_df)} rows to {landscape_output}")
    print(f"Wrote {len(iterates_df)} rows to {iterate_output}")


if __name__ == "__main__":
    main()
