import pytest
import torch

from decision_learning.modeling.loss import (
    MSELoss,
    CosineEmbeddingLoss,
    PGLoss,
    SPOPlusLoss,
    FYLoss,
    CosineSurrogateDotProdMSELoss,
    CosineSurrogateDotProdVecMagLoss,
    PGDCALoss,
    PGAdaptiveLoss,
    CILOLoss,
)
from decision_learning.modeling.smoothing import RandomizedSmoothingWrapper
from decision_learning.utils import filter_kwargs

SUPPORTED_LOSSES = [
    MSELoss,
    CosineEmbeddingLoss,
    PGLoss,
    SPOPlusLoss,
    FYLoss,
    CosineSurrogateDotProdMSELoss,
    CosineSurrogateDotProdVecMagLoss,
    PGDCALoss,
    PGAdaptiveLoss,
    CILOLoss,
]


def _box_oracle_torch(costs: torch.Tensor, b: torch.Tensor):
    """Solve min <c, z> s.t. 0 <= z <= b, rowwise."""
    if costs.dim() == 1:
        costs = costs.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    z = b * (costs <= 0).to(costs.dtype)
    obj = torch.sum(costs * z, dim=1, keepdim=True)
    return z, obj


def _box_oracle_with_kwargs(costs: torch.Tensor, **instance_kwargs):
    if costs.dtype != torch.float32:
        costs = costs.to(torch.float32)
    return _box_oracle_torch(costs, instance_kwargs["b"])


def _sum_obj_oracle(costs: torch.Tensor, **instance_kwargs):
    if costs.dim() == 1:
        costs = costs.unsqueeze(0)
    z = torch.ones_like(costs)
    obj = torch.sum(costs * z, dim=1, keepdim=True)
    return z, obj


def _make_standard_batch():
    batch_size = 4
    obs_cost = torch.tensor(
        [[-1.0, 2.0, -0.5], [0.5, -1.0, 1.5], [-2.0, 0.1, 0.2], [1.0, -0.2, -0.3]],
        dtype=torch.float32,
    )
    instance_kwargs = {"b": torch.ones_like(obs_cost)}
    obs_sol, obs_obj = _box_oracle_torch(obs_cost, instance_kwargs["b"])

    batch = {
        "obs_cost": obs_cost,
        "obs_sol": obs_sol,
        "obs_obj": obs_obj,
        "instance_kwargs": instance_kwargs,
        "unused_extra": torch.zeros(batch_size),
    }
    return obs_cost, batch


def _assert_loss_backward(loss_fn, pred, batch):
    filtered = filter_kwargs(loss_fn.forward, batch)
    loss = loss_fn(pred, **filtered)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()


def test_pg_dca_loss_contract_backward_step():
    torch.manual_seed(0)
    obs_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    input_dim = 4
    pred_model = torch.nn.Linear(input_dim, obs_cost.shape[1])
    X = torch.randn(obs_cost.shape[0], input_dim)
    pred_cost = pred_model(X)

    loss_fn = PGDCALoss(
        optmodel=_box_oracle_with_kwargs,
        h=0.1,
        reduction="mean",
        is_minimization=True,
        update_every=1,
    )

    loss = loss_fn(
        pred_cost,
        X=X,
        obs_cost=obs_cost,
        pred_model=pred_model,
        instance_kwargs=instance_kwargs,
    )
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    loss.backward()
    assert pred_model.weight.grad is not None
    assert torch.isfinite(pred_model.weight.grad).all()


def _build_linear_pred_cost(obs_cost: torch.Tensor, seed: int = 0):
    torch.manual_seed(seed)
    input_dim = 4
    pred_model = torch.nn.Linear(input_dim, obs_cost.shape[1])
    X = torch.randn(obs_cost.shape[0], input_dim)
    pred_cost = pred_model(X)
    return pred_model, X, pred_cost


def _build_loss_inputs_for_reduction(loss_cls, obs_cost, batch, seed: int = 0):
    if loss_cls in (PGDCALoss, PGAdaptiveLoss, CILOLoss):
        pred_model, X, pred_cost = _build_linear_pred_cost(obs_cost, seed=seed)
        batch = dict(batch)
        if loss_cls in (PGDCALoss, PGAdaptiveLoss):
            batch["X"] = X
            batch["pred_model"] = pred_model
        pred = pred_cost

        if loss_cls is PGDCALoss:
            def _factory(reduction: str):
                return loss_cls(
                    optmodel=_box_oracle_with_kwargs,
                    h=0.1,
                    reduction=reduction,
                    is_minimization=True,
                    update_every=1,
                )
        elif loss_cls is PGAdaptiveLoss:
            def _factory(reduction: str):
                return loss_cls(
                    optmodel=_box_oracle_with_kwargs,
                    beta=0.1,
                    reduction=reduction,
                    is_minimization=True,
                )
        else:
            def _factory(reduction: str):
                return loss_cls(
                    optmodel=_box_oracle_with_kwargs,
                    beta=0.1,
                    reduction=reduction,
                    is_minimization=True,
                )
        return pred, batch, _factory

    if loss_cls in (PGLoss, SPOPlusLoss, FYLoss):
        pred = (obs_cost + 0.05 * torch.randn_like(obs_cost)).requires_grad_(True)

        def _factory(reduction: str):
            return loss_cls(
                optmodel=_box_oracle_with_kwargs,
                reduction=reduction,
                is_minimization=True,
                **({"h": 0.1, "finite_diff_type": "B"} if loss_cls is PGLoss else {}),
            )
        return pred, batch, _factory

    pred = (obs_cost + 0.05 * torch.randn_like(obs_cost)).requires_grad_(True)
    if loss_cls in (CosineSurrogateDotProdMSELoss, CosineSurrogateDotProdVecMagLoss):
        def _factory(reduction: str):
            return loss_cls(alpha=0.5, reduction=reduction, is_minimization=True)
    else:
        def _factory(reduction: str):
            return loss_cls(reduction=reduction)
    return pred, batch, _factory


@pytest.mark.parametrize("loss_cls", SUPPORTED_LOSSES)
def test_loss_per_sample_matches_mean(loss_cls):
    torch.manual_seed(0)
    obs_cost, batch = _make_standard_batch()

    pred, batch, factory = _build_loss_inputs_for_reduction(loss_cls, obs_cost, batch, seed=4)

    loss_fn = factory("mean")
    loss_per_sample = loss_fn.per_sample(pred, **batch)
    assert torch.is_tensor(loss_per_sample)
    assert loss_per_sample.ndim in (1, 2)
    if loss_per_sample.ndim == 2:
        assert loss_per_sample.shape[1] == 1

    loss_mean = loss_fn(pred, **batch)
    assert torch.is_tensor(loss_mean)
    assert loss_mean.ndim == 0
    assert torch.isfinite(loss_mean).all()

    assert torch.allclose(loss_per_sample.mean(), loss_mean, rtol=1e-5, atol=1e-5)


def test_pg_loss_adaptive_contract_backward_step():
    torch.manual_seed(0)
    obs_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    pred_model, X, pred_cost = _build_linear_pred_cost(obs_cost, seed=1)

    loss_fn = PGAdaptiveLoss(
        optmodel=_box_oracle_with_kwargs,
        beta=0.1,
        reduction="mean",
        is_minimization=True,
    )

    loss = loss_fn(
        pred_cost,
        X=X,
        obs_cost=obs_cost,
        pred_model=pred_model,
        instance_kwargs=instance_kwargs,
    )
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    loss.backward()
    assert pred_model.weight.grad is not None
    assert torch.isfinite(pred_model.weight.grad).all()


def test_pg_loss_scale_by_norm_central_diff():
    torch.manual_seed(0)
    obs_cost = torch.tensor(
        [[3.0, 4.0], [0.0, 0.0], [-3.0, 0.0]],
        dtype=torch.float32,
    )
    pred_cost = torch.zeros_like(obs_cost)

    loss_fn_scaled = PGLoss(
        optmodel=_sum_obj_oracle,
        h=0.5,
        finite_diff_type="C",
        reduction="none",
        is_minimization=True,
        scale_by_norm=True,
    )
    loss_fn_unscaled = PGLoss(
        optmodel=_sum_obj_oracle,
        h=0.5,
        finite_diff_type="C",
        reduction="none",
        is_minimization=True,
        scale_by_norm=False,
    )

    scaled_loss = loss_fn_scaled.per_sample(pred_cost, obs_cost=obs_cost)
    unscaled_loss = loss_fn_unscaled.per_sample(pred_cost, obs_cost=obs_cost)

    expected_scaled = torch.tensor([1.4, 0.0, -1.0], dtype=torch.float32)
    expected_unscaled = torch.tensor([7.0, 0.0, -3.0], dtype=torch.float32)

    assert torch.allclose(scaled_loss, expected_scaled, rtol=1e-6, atol=1e-6)
    assert torch.allclose(unscaled_loss, expected_unscaled, rtol=1e-6, atol=1e-6)


def test_cilo_loss_contract_backward_step():
    torch.manual_seed(0)
    obs_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    pred_model, _, pred_cost = _build_linear_pred_cost(obs_cost, seed=2)

    loss_fn = CILOLoss(
        optmodel=_box_oracle_with_kwargs,
        beta=0.1,
        reduction="mean",
        is_minimization=True,
    )

    loss = loss_fn(
        pred_cost,
        obs_cost=obs_cost,
        instance_kwargs=instance_kwargs,
    )
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    loss.backward()
    assert pred_model.weight.grad is not None
    assert torch.isfinite(pred_model.weight.grad).all()


@pytest.mark.parametrize("loss_cls", SUPPORTED_LOSSES)
def test_loss_standard_signature_acceptance(loss_cls):
    torch.manual_seed(0)
    obs_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    if loss_cls in (PGDCALoss, PGAdaptiveLoss, CILOLoss):
        pred_model, X, pred_cost = _build_linear_pred_cost(obs_cost, seed=3)
        batch = dict(batch)
        if loss_cls in (PGDCALoss, PGAdaptiveLoss):
            batch["X"] = X
            batch["pred_model"] = pred_model
        pred = pred_cost
        if loss_cls is PGDCALoss:
            loss_fn = loss_cls(
                optmodel=_box_oracle_with_kwargs,
                h=0.1,
                reduction="mean",
                is_minimization=True,
                update_every=1,
            )
        elif loss_cls is PGAdaptiveLoss:
            loss_fn = loss_cls(
                optmodel=_box_oracle_with_kwargs,
                beta=0.1,
                reduction="mean",
                is_minimization=True,
            )
        else:
            loss_fn = loss_cls(
                optmodel=_box_oracle_with_kwargs,
                beta=0.1,
                reduction="mean",
                is_minimization=True,
            )
    elif loss_cls in (PGLoss, SPOPlusLoss, FYLoss):
        pred = (obs_cost + 0.05 * torch.randn_like(obs_cost)).requires_grad_(True)
        loss_fn = loss_cls(
            optmodel=_box_oracle_with_kwargs,
            reduction="mean",
            is_minimization=True,
            **({"h": 0.1, "finite_diff_type": "B"} if loss_cls is PGLoss else {}),
        )
    else:
        pred = (obs_cost + 0.05 * torch.randn_like(obs_cost)).requires_grad_(True)
        if loss_cls in (CosineSurrogateDotProdMSELoss, CosineSurrogateDotProdVecMagLoss):
            loss_fn = loss_cls(alpha=0.5, reduction="mean", is_minimization=True)
        else:
            loss_fn = loss_cls(reduction="mean")

    loss = loss_fn(pred, **batch)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()

    if loss_cls not in (PGDCALoss, PGAdaptiveLoss, CILOLoss):
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("loss_cls", [PGDCALoss, PGAdaptiveLoss, CILOLoss])
def test_pred_cost_grad_for_pg_family_losses(loss_cls):
    torch.manual_seed(0)
    obs_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    input_dim = 4
    pred_model = torch.nn.Linear(input_dim, obs_cost.shape[1])
    X = torch.randn(obs_cost.shape[0], input_dim)
    pred_cost = pred_model(X).detach().requires_grad_(True)

    if loss_cls is PGDCALoss:
        loss_fn = loss_cls(
            optmodel=_box_oracle_with_kwargs,
            h=0.1,
            reduction="mean",
            is_minimization=True,
            update_every=1,
        )
    elif loss_cls is PGAdaptiveLoss:
        loss_fn = loss_cls(
            optmodel=_box_oracle_with_kwargs,
            beta=0.1,
            reduction="mean",
            is_minimization=True,
        )
    else:
        loss_fn = loss_cls(
            optmodel=_box_oracle_with_kwargs,
            beta=0.1,
            reduction="mean",
            is_minimization=True,
        )

    loss = loss_fn(
        pred_cost,
        obs_cost=obs_cost,
        instance_kwargs=instance_kwargs,
        **({"X": X, "pred_model": pred_model} if loss_cls in (PGDCALoss, PGAdaptiveLoss) else {}),
    )
    loss.backward()
    assert pred_cost.grad is not None
    assert torch.isfinite(pred_cost.grad).all()


def test_spoplus_smoothing_path_backward():
    torch.manual_seed(0)
    obs_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    pred = (obs_cost + 0.05 * torch.randn_like(obs_cost)).requires_grad_(True)
    base_loss = SPOPlusLoss(
        optmodel=_box_oracle_with_kwargs,
        reduction="mean",
        is_minimization=True,
    )
    loss_fn = RandomizedSmoothingWrapper(
        base_loss=base_loss,
        sigma=0.1,
        s=2,
        seed=123,
        antithetic=False,
        control_variate=False,
        reduction="mean",
    )

    loss = loss_fn(
        pred_cost=pred,
        obs_cost=obs_cost,
        obs_sol=batch["obs_sol"],
        obs_obj=batch["obs_obj"],
        instance_kwargs=instance_kwargs,
    )
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
