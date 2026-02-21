from __future__ import annotations

import argparse

import pandas as pd
import torch

from decision_learning.benchmarks.hinge.data import genData
from decision_learning.benchmarks.hinge.oracle import opt_oracle
from decision_learning.modeling.loss import PGLoss
from decision_learning.modeling.smoothing import RandomizedSmoothingWrapper


# ----------------------- Global parameters -----------------------
h = 1.0
sigma = 1.0
s = 30
antithetic = False
control_variate = False
num_datasets = 100
num_data = 100
test_point = 1.0
step_size = 0.01
base_seed = 135


def _make_losses() -> dict[str, torch.nn.Module]:
    pg_raw = PGLoss(
        optmodel=opt_oracle,
        h=h,
        finite_diff_type="B",
        reduction="mean",
        is_minimization=True,
    )

    # Separate base-loss objects keep each wrapper independent.
    pg_for_smooth_base = PGLoss(optmodel=opt_oracle, h=h, finite_diff_type="B", reduction="none")
    pg_for_smooth_sigma = PGLoss(optmodel=opt_oracle, h=h, finite_diff_type="B", reduction="none")
    pg_for_smooth_s = PGLoss(optmodel=opt_oracle, h=h, finite_diff_type="B", reduction="none")
    pg_for_smooth_antithetic = PGLoss(optmodel=opt_oracle, h=h, finite_diff_type="B", reduction="none")
    pg_for_smooth_control = PGLoss(optmodel=opt_oracle, h=h, finite_diff_type="B", reduction="none")

    smooth_base = RandomizedSmoothingWrapper(
        base_loss=pg_for_smooth_base,
        sigma=sigma,
        s=s,
        antithetic=antithetic,
        control_variate=control_variate,
        reduction="mean",
    )
    smooth_sigma = RandomizedSmoothingWrapper(
        base_loss=pg_for_smooth_sigma,
        sigma=0.1 * sigma,
        s=s,
        antithetic=antithetic,
        control_variate=control_variate,
        reduction="mean",
    )
    smooth_s = RandomizedSmoothingWrapper(
        base_loss=pg_for_smooth_s,
        sigma=sigma,
        s=5 * s,
        antithetic=antithetic,
        control_variate=control_variate,
        reduction="mean",
    )
    smooth_antithetic = RandomizedSmoothingWrapper(
        base_loss=pg_for_smooth_antithetic,
        sigma=sigma,
        s=s,
        antithetic=True,
        control_variate=control_variate,
        reduction="mean",
    )
    smooth_control = RandomizedSmoothingWrapper(
        base_loss=pg_for_smooth_control,
        sigma=sigma,
        s=s,
        antithetic=antithetic,
        control_variate=True,
        reduction="mean",
    )

    return {
        "pg": pg_raw,
        "smooth_base": smooth_base,
        "smooth_sigma": smooth_sigma,
        "smooth_s": smooth_s,
        "smooth_antithetic": smooth_antithetic,
        "smooth_control": smooth_control,
    }


def _in_sample_loss_at_point(loss_fn: torch.nn.Module, obs_cost: torch.Tensor, point: float) -> float:
    pred_cost = torch.full_like(obs_cost, fill_value=point)
    out = loss_fn(pred_cost=pred_cost, obs_cost=obs_cost)
    return float(out.item())


def _central_diff_grad(
    loss_fn: torch.nn.Module,
    obs_cost: torch.Tensor,
    point: float,
    delta: float,
) -> float:
    loss_plus = _in_sample_loss_at_point(loss_fn, obs_cost, point + delta)
    loss_minus = _in_sample_loss_at_point(loss_fn, obs_cost, point - delta)
    return (loss_plus - loss_minus) / (2.0 * delta)


def _autograd_grad_wrt_scalar_point(
    loss_fn: torch.nn.Module,
    obs_cost: torch.Tensor,
    point: float,
) -> float:
    pred_cost = torch.full_like(obs_cost, fill_value=point, requires_grad=True)
    loss = loss_fn(pred_cost=pred_cost, obs_cost=obs_cost)
    loss.backward()
    # pred_cost entries all come from a single scalar point, so use total derivative.
    return float(pred_cost.grad.sum().item())


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    losses = _make_losses()
    rows: list[dict[str, float]] = []

    for i in range(num_datasets):
        dataset = genData(num_data=num_data, seed=base_seed + i)
        obs_cost = torch.tensor(dataset["cost"], dtype=torch.float32)

        row: dict[str, float] = {}
        for loss_name, loss_fn in losses.items():
            _ = _in_sample_loss_at_point(loss_fn, obs_cost, test_point)
            if loss_name == "pg":
                row[loss_name] = _central_diff_grad(loss_fn, obs_cost, test_point, step_size)
            else:
                row[loss_name] = _autograd_grad_wrt_scalar_point(loss_fn, obs_cost, test_point)
        rows.append(row)

    grad_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame({"mean": grad_df.mean(), "std": grad_df.std(ddof=1)})
    return grad_df, summary_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PG loss vs randomized smoothing variants on hinge benchmark.",
    )
    parser.add_argument("--num-datasets", type=int, default=num_datasets)
    parser.add_argument("--num-data", type=int, default=num_data)
    parser.add_argument("--test-point", type=float, default=test_point)
    parser.add_argument("--step-size", type=float, default=step_size)
    parser.add_argument("--h", type=float, default=h)
    parser.add_argument("--sigma", type=float, default=sigma)
    parser.add_argument("--s", type=int, default=s)
    parser.add_argument("--base-seed", type=int, default=base_seed)
    return parser.parse_args()


def main() -> None:
    global num_datasets, num_data, test_point, step_size, h, sigma, s, base_seed

    args = _parse_args()
    num_datasets = args.num_datasets
    num_data = args.num_data
    test_point = args.test_point
    step_size = args.step_size
    h = args.h
    sigma = args.sigma
    s = args.s
    base_seed = args.base_seed

    grad_df, summary_df = run_experiment()

    # print("\nGradient estimates per dataset (rows) and loss (columns):")
    # print(grad_df.to_string(index=False))

    print("\nMean and standard deviation by loss:")
    print(summary_df.to_string(float_format=lambda x: f"{x: .6f}"))


if __name__ == "__main__":
    main()
