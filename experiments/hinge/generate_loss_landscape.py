from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from decision_learning.benchmarks.hinge.data import genData
from decision_learning.benchmarks.hinge.oracle import opt_oracle
from decision_learning.modeling.loss import CILOLoss, FYLoss, MSELoss, PGDCALoss, PGLoss, SPOPlusLoss
from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.smoothing import RandomizedSmoothingWrapper
from decision_learning.utils import handle_solver


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _scale_to_unit_interval(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    xmin = np.min(arr)
    xmax = np.max(arr)
    if xmax == xmin:
        return np.zeros_like(arr, dtype=float)
    return (arr - xmin) / (xmax - xmin)


def _decision_loss(pred_cost: torch.Tensor, cost: torch.Tensor, optmodel: callable) -> torch.Tensor:
    pred_sol, _ = optmodel(pred_cost)
    pred_sol = torch.as_tensor(pred_sol, dtype=cost.dtype, device=cost.device)
    aligned_cost = torch.as_tensor(cost, dtype=cost.dtype, device=cost.device)

    if pred_sol.ndim == 1:
        pred_sol = pred_sol.reshape(-1, 1)
    if aligned_cost.ndim == 1:
        aligned_cost = aligned_cost.reshape(-1, 1)

    return (pred_sol * aligned_cost).sum(dim=1, keepdim=True)


def _make_smooth_loss(
    base_loss: torch.nn.Module,
    sigma: float,
    s: int,
    antithetic: bool,
    control_variate: bool,
) -> RandomizedSmoothingWrapper:
    return RandomizedSmoothingWrapper(
        base_loss=base_loss,
        sigma=sigma,
        s=s,
        seed=42,
        antithetic=antithetic,
        control_variate=control_variate,
        reduction="mean",
    )


def _make_linear_model(input_dim: int, output_dim: int, weight: torch.Tensor, bias: torch.Tensor) -> LinearRegression:
    model = LinearRegression(input_dim=input_dim, output_dim=output_dim)
    with torch.no_grad():
        model.linear.weight.copy_(weight)
        model.linear.bias.copy_(bias.reshape(-1))
    return model


def _fit_mse_linear_model(X: torch.Tensor, obs_cost: torch.Tensor) -> LinearRegression:
    ones = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
    design = torch.cat([X, ones], dim=1)
    solution = torch.linalg.lstsq(design, obs_cost).solution
    weight = solution[:-1, :].T.contiguous()
    bias = solution[-1, :].contiguous()
    return _make_linear_model(
        input_dim=X.shape[1],
        output_dim=obs_cost.shape[1],
        weight=weight,
        bias=bias,
    )


def _build_losses(
    *,
    optmodel: callable,
    h: float,
    sigma: float,
    s: int,
    antithetic: bool,
    control_variate: bool,
    beta: float,
    model0: LinearRegression,
) -> dict[str, torch.nn.Module]:
    pgb = PGLoss(optmodel=optmodel, h=h, finite_diff_type="B")
    pgc = PGLoss(optmodel=optmodel, h=h, finite_diff_type="C")
    spoplus = SPOPlusLoss(optmodel=optmodel)
    fy = FYLoss(optmodel=optmodel)

    return {
        "pgb_loss": pgb,
        "pgc_loss": pgc,
        "mse_loss": MSELoss(),
        "spoplus_loss": spoplus,
        "fy_loss": fy,
        "cilo_loss": CILOLoss(optmodel=optmodel, beta=beta),
        "pgdca_loss": PGDCALoss(optmodel=optmodel, h=h, update_every=0, model0=model0),
        "smooth_pgb_loss": _make_smooth_loss(
            base_loss=PGLoss(optmodel=optmodel, h=h, finite_diff_type="B"),
            sigma=sigma,
            s=s,
            antithetic=antithetic,
            control_variate=control_variate,
        ),
        "smooth_pgc_loss": _make_smooth_loss(
            base_loss=PGLoss(optmodel=optmodel, h=h, finite_diff_type="C"),
            sigma=sigma,
            s=s,
            antithetic=antithetic,
            control_variate=control_variate,
        ),
        "smooth_spoplus_loss": _make_smooth_loss(
            base_loss=SPOPlusLoss(optmodel=optmodel),
            sigma=sigma,
            s=s,
            antithetic=antithetic,
            control_variate=control_variate,
        ),
        "smooth_fy_loss": _make_smooth_loss(
            base_loss=FYLoss(optmodel=optmodel),
            sigma=sigma,
            s=s,
            antithetic=antithetic,
            control_variate=control_variate,
        ),
    }


def _evaluate_losses(
    *,
    betas: np.ndarray,
    X: torch.Tensor,
    intercept: float,
    obs_cost: torch.Tensor,
    cond_exp_cost: torch.Tensor,
    obs_sol: torch.Tensor,
    obs_obj: torch.Tensor,
    losses: dict[str, torch.nn.Module],
    optmodel: callable,
    input_dim: int,
    output_dim: int,
) -> dict[str, np.ndarray]:
    results: dict[str, list[float]] = {
        "decision_loss": [],
        "oracle_decision_loss": [],
    }
    for loss_name in losses:
        results[loss_name] = []

    for beta in betas:
        pred_cost = beta * X + intercept
        pred_model = _make_linear_model(
            input_dim=input_dim,
            output_dim=output_dim,
            weight=torch.tensor([[beta]], dtype=X.dtype, device=X.device),
            bias=torch.tensor([intercept], dtype=X.dtype, device=X.device),
        )

        results["decision_loss"].append(_decision_loss(pred_cost, obs_cost, optmodel).mean().item())
        results["oracle_decision_loss"].append(_decision_loss(pred_cost, cond_exp_cost, optmodel).mean().item())

        for loss_name, loss_fn in losses.items():
            loss_val = loss_fn(
                pred_cost,
                obs_cost=obs_cost,
                obs_sol=obs_sol,
                obs_obj=obs_obj,
                X=X,
                pred_model=pred_model,
            ).item()
            results[loss_name].append(loss_val)

    return {name: _scale_to_unit_interval(values) for name, values in results.items()}


def generate_loss_landscape(args: argparse.Namespace) -> pd.DataFrame:
    data = genData(num_data=args.num_data, m=args.m, alpha=args.alpha, seed=args.seed)

    X = torch.tensor(data["feat"], dtype=torch.float32)
    cond_exp_cost = torch.tensor(data["cond_exp_cost"], dtype=torch.float32)
    obs_cost = torch.tensor(data["cost"], dtype=torch.float32)

    optmodel = partial(handle_solver, optmodel=opt_oracle, detach_tensor=False, solver_batch_solve=True)
    obs_sol, obs_obj = optmodel(obs_cost)
    model0 = _fit_mse_linear_model(X, obs_cost)

    losses = _build_losses(
        optmodel=optmodel,
        h=args.h,
        sigma=args.sigma,
        s=args.s,
        antithetic=args.antithetic,
        control_variate=args.control_variate,
        beta=obs_obj.mean().item() * args.cilo_beta_scale,
        model0=model0,
    )

    betas = np.linspace(args.beta_min, args.beta_max, args.num_beta_points)
    scaled_results = _evaluate_losses(
        betas=betas,
        X=X,
        intercept=args.intercept,
        obs_cost=obs_cost,
        cond_exp_cost=cond_exp_cost,
        obs_sol=obs_sol,
        obs_obj=obs_obj,
        losses=losses,
        optmodel=optmodel,
        input_dim=X.shape[1],
        output_dim=obs_cost.shape[1],
    )

    return pd.DataFrame({"beta": betas, **scaled_results})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the hinge loss landscape CSV used by HingeLossLandscape.ipynb.",
    )
    parser.add_argument("--num-data", type=int, default=200)
    parser.add_argument("--m", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=885309)
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=10.0)
    parser.add_argument("--num-beta-points", type=int, default=101)
    parser.add_argument("--intercept", type=float, default=-2.0)
    parser.add_argument("--h", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--s", type=int, default=5)
    parser.add_argument("--antithetic", dest="antithetic", action="store_true")
    parser.add_argument("--no-antithetic", dest="antithetic", action="store_false")
    parser.add_argument("--control-variate", action="store_true", default=False)
    parser.add_argument("--cilo-beta-scale", type=float, default=0.9)
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_root() / "outputs" / "hinge" / "hinge_loss_landscape.csv",
    )
    parser.set_defaults(antithetic=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = generate_loss_landscape(args)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
