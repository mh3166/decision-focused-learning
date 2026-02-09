import pytest
import torch

from decision_learning.modeling.loss import (
    MSELoss,
    CosineEmbeddingLoss,
    PG_Loss,
    SPOPlus,
    perturbedFenchelYoung,
    CosineSurrogateDotProdMSE,
    CosineSurrogateDotProdVecMag,
    PG_DCA_Loss,
    PG_Loss_Adaptive,
    CILO_Loss,
)
from decision_learning.utils import filter_kwargs

SUPPORTED_LOSSES = [
    MSELoss,
    CosineEmbeddingLoss,
    PG_Loss,
    SPOPlus,
    perturbedFenchelYoung,
    CosineSurrogateDotProdMSE,
    CosineSurrogateDotProdVecMag,
    PG_DCA_Loss,
    PG_Loss_Adaptive,
    CILO_Loss,
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


def _make_standard_batch():
    batch_size = 4
    true_cost = torch.tensor(
        [[-1.0, 2.0, -0.5], [0.5, -1.0, 1.5], [-2.0, 0.1, 0.2], [1.0, -0.2, -0.3]],
        dtype=torch.float32,
    )
    instance_kwargs = {"b": torch.ones_like(true_cost)}
    true_sol, true_obj = _box_oracle_torch(true_cost, instance_kwargs["b"])

    batch = {
        "true_cost": true_cost,
        "true_sol": true_sol,
        "true_obj": true_obj,
        "instance_kwargs": instance_kwargs,
        "unused_extra": torch.zeros(batch_size),
    }
    return true_cost, batch


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
    true_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    input_dim = 4
    pred_model = torch.nn.Linear(input_dim, true_cost.shape[1])
    X = torch.randn(true_cost.shape[0], input_dim)
    pred_cost = pred_model(X)

    loss_fn = PG_DCA_Loss(
        optmodel=_box_oracle_with_kwargs,
        h=0.1,
        reduction="mean",
        minimize=True,
        update_every=1,
    )

    loss = loss_fn(
        pred_cost,
        X=X,
        true_cost=true_cost,
        pred_model=pred_model,
        instance_kwargs=instance_kwargs,
    )
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    loss.backward()
    assert pred_model.weight.grad is not None
    assert torch.isfinite(pred_model.weight.grad).all()


def _build_linear_pred_cost(true_cost: torch.Tensor, seed: int = 0):
    torch.manual_seed(seed)
    input_dim = 4
    pred_model = torch.nn.Linear(input_dim, true_cost.shape[1])
    X = torch.randn(true_cost.shape[0], input_dim)
    pred_cost = pred_model(X)
    return pred_model, X, pred_cost


def _build_loss_inputs_for_reduction(loss_cls, true_cost, batch, seed: int = 0):
    if loss_cls is perturbedFenchelYoung:
        pytest.xfail("FY perturbed path does not yet expand instance_kwargs to match n_samples * batch.")

    if loss_cls in (PG_DCA_Loss, PG_Loss_Adaptive, CILO_Loss):
        pred_model, X, pred_cost = _build_linear_pred_cost(true_cost, seed=seed)
        batch = dict(batch)
        batch["X"] = X
        batch["pred_model"] = pred_model
        pred = pred_cost

        if loss_cls is PG_DCA_Loss:
            def _factory(reduction: str):
                return loss_cls(
                    optmodel=_box_oracle_with_kwargs,
                    h=0.1,
                    reduction=reduction,
                    minimize=True,
                    update_every=1,
                )
        elif loss_cls is PG_Loss_Adaptive:
            def _factory(reduction: str):
                return loss_cls(
                    optmodel=_box_oracle_with_kwargs,
                    beta=0.1,
                    reduction=reduction,
                    minimize=True,
                )
        else:
            def _factory(reduction: str):
                return loss_cls(
                    optmodel=_box_oracle_with_kwargs,
                    beta=0.1,
                    reduction=reduction,
                    minimize=True,
                )
        return pred, batch, _factory

    if loss_cls in (PG_Loss, SPOPlus):
        pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)

        def _factory(reduction: str):
            return loss_cls(
                optmodel=_box_oracle_with_kwargs,
                reduction=reduction,
                minimize=True,
                **({"h": 0.1, "finite_diff_type": "B"} if loss_cls is PG_Loss else {}),
            )
        return pred, batch, _factory

    if loss_cls is perturbedFenchelYoung:
        pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)

        def _factory(reduction: str):
            return loss_cls(
                optmodel=_box_oracle_with_kwargs,
                n_samples=3,
                sigma=0.1,
                seed=123,
                reduction=reduction,
                minimize=True,
            )
        return pred, batch, _factory

    pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)
    if loss_cls in (CosineSurrogateDotProdMSE, CosineSurrogateDotProdVecMag):
        def _factory(reduction: str):
            return loss_cls(alpha=0.5, reduction=reduction, minimize=True)
    else:
        def _factory(reduction: str):
            return loss_cls(reduction=reduction)
    return pred, batch, _factory


@pytest.mark.parametrize("loss_cls", SUPPORTED_LOSSES)
def test_loss_per_sample_matches_mean(loss_cls):
    torch.manual_seed(0)
    true_cost, batch = _make_standard_batch()

    pred, batch, factory = _build_loss_inputs_for_reduction(loss_cls, true_cost, batch, seed=4)

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
    true_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    pred_model, X, pred_cost = _build_linear_pred_cost(true_cost, seed=1)

    loss_fn = PG_Loss_Adaptive(
        optmodel=_box_oracle_with_kwargs,
        beta=0.1,
        reduction="mean",
        minimize=True,
    )

    loss = loss_fn(
        pred_cost,
        X=X,
        true_cost=true_cost,
        pred_model=pred_model,
        instance_kwargs=instance_kwargs,
    )
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    loss.backward()
    assert pred_model.weight.grad is not None
    assert torch.isfinite(pred_model.weight.grad).all()


def test_cilo_loss_contract_backward_step():
    torch.manual_seed(0)
    true_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    pred_model, X, pred_cost = _build_linear_pred_cost(true_cost, seed=2)

    loss_fn = CILO_Loss(
        optmodel=_box_oracle_with_kwargs,
        beta=0.1,
        reduction="mean",
        minimize=True,
    )

    loss = loss_fn(
        pred_cost,
        X=X,
        true_cost=true_cost,
        pred_model=pred_model,
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
    true_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    if loss_cls is perturbedFenchelYoung:
        pytest.xfail("FY perturbed path does not yet expand instance_kwargs to match n_samples * batch.")

    if loss_cls in (PG_DCA_Loss, PG_Loss_Adaptive, CILO_Loss):
        pred_model, X, pred_cost = _build_linear_pred_cost(true_cost, seed=3)
        batch = dict(batch)
        batch["X"] = X
        batch["pred_model"] = pred_model
        pred = pred_cost
        if loss_cls is PG_DCA_Loss:
            loss_fn = loss_cls(
                optmodel=_box_oracle_with_kwargs,
                h=0.1,
                reduction="mean",
                minimize=True,
                update_every=1,
            )
        elif loss_cls is PG_Loss_Adaptive:
            loss_fn = loss_cls(
                optmodel=_box_oracle_with_kwargs,
                beta=0.1,
                reduction="mean",
                minimize=True,
            )
        else:
            loss_fn = loss_cls(
                optmodel=_box_oracle_with_kwargs,
                beta=0.1,
                reduction="mean",
                minimize=True,
            )
    elif loss_cls in (PG_Loss, SPOPlus):
        pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)
        loss_fn = loss_cls(
            optmodel=_box_oracle_with_kwargs,
            reduction="mean",
            minimize=True,
            **({"h": 0.1, "finite_diff_type": "B"} if loss_cls is PG_Loss else {}),
        )
    elif loss_cls is perturbedFenchelYoung:
        pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)
        loss_fn = loss_cls(
            optmodel=_box_oracle_with_kwargs,
            n_samples=3,
            sigma=0.1,
            seed=123,
            reduction="mean",
            minimize=True,
        )
    else:
        pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)
        if loss_cls in (CosineSurrogateDotProdMSE, CosineSurrogateDotProdVecMag):
            loss_fn = loss_cls(alpha=0.5, reduction="mean", minimize=True)
        else:
            loss_fn = loss_cls(reduction="mean")

    loss = loss_fn(pred, **batch)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()

    if loss_cls not in (PG_DCA_Loss, PG_Loss_Adaptive, CILO_Loss):
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize("loss_cls", [PG_DCA_Loss, PG_Loss_Adaptive, CILO_Loss])
def test_pred_cost_grad_for_pg_family_losses(loss_cls):
    torch.manual_seed(0)
    true_cost, batch = _make_standard_batch()
    instance_kwargs = batch["instance_kwargs"]

    input_dim = 4
    pred_model = torch.nn.Linear(input_dim, true_cost.shape[1])
    X = torch.randn(true_cost.shape[0], input_dim)
    pred_cost = pred_model(X).detach().requires_grad_(True)

    if loss_cls is PG_DCA_Loss:
        loss_fn = loss_cls(
            optmodel=_box_oracle_with_kwargs,
            h=0.1,
            reduction="mean",
            minimize=True,
            update_every=1,
        )
    elif loss_cls is PG_Loss_Adaptive:
        loss_fn = loss_cls(
            optmodel=_box_oracle_with_kwargs,
            beta=0.1,
            reduction="mean",
            minimize=True,
        )
    else:
        loss_fn = loss_cls(
            optmodel=_box_oracle_with_kwargs,
            beta=0.1,
            reduction="mean",
            minimize=True,
        )

    loss = loss_fn(
        pred_cost,
        X=X,
        true_cost=true_cost,
        pred_model=pred_model,
        instance_kwargs=instance_kwargs,
    )
    loss.backward()
    assert pred_cost.grad is not None
    assert torch.isfinite(pred_cost.grad).all()
