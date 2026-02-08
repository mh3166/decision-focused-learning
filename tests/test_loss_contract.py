import pytest
import torch

from decision_learning.modeling.loss import (
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


def test_pg_loss_contract_backward_step():
    torch.manual_seed(0)
    true_cost, batch = _make_standard_batch()
    pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)

    loss_fn = PG_Loss(
        optmodel=_box_oracle_with_kwargs,
        h=0.1,
        finite_diff_type="B",
        reduction="mean",
        minimize=True,
    )
    _assert_loss_backward(loss_fn, pred, batch)


def test_spoplus_loss_contract_backward_step():
    torch.manual_seed(0)
    true_cost, batch = _make_standard_batch()
    pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)

    loss_fn = SPOPlus(
        optmodel=_box_oracle_with_kwargs,
        reduction="mean",
        minimize=True,
    )
    _assert_loss_backward(loss_fn, pred, batch)


@pytest.mark.xfail(reason="FY perturbed path does not yet expand instance_kwargs to match n_samples * batch.")
def test_perturbed_fenchel_young_loss_contract_backward_step():
    torch.manual_seed(0)
    true_cost, batch = _make_standard_batch()
    pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)

    loss_fn = perturbedFenchelYoung(
        optmodel=_box_oracle_with_kwargs,
        n_samples=3,
        sigma=0.1,
        seed=123,
        reduction="mean",
        minimize=True,
    )
    _assert_loss_backward(loss_fn, pred, batch)


def test_cosine_surrogate_dotprod_mse_contract_backward_step():
    torch.manual_seed(0)
    true_cost, batch = _make_standard_batch()
    pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)

    loss_fn = CosineSurrogateDotProdMSE(alpha=0.5, reduction="mean", minimize=True)
    _assert_loss_backward(loss_fn, pred, batch)


def test_cosine_surrogate_vecmag_contract_backward_step():
    torch.manual_seed(0)
    true_cost, batch = _make_standard_batch()
    pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)

    loss_fn = CosineSurrogateDotProdVecMag(alpha=0.5, reduction="mean", minimize=True)
    _assert_loss_backward(loss_fn, pred, batch)


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


@pytest.mark.xfail(reason="MSELoss expects target, standardized batch does not include it yet.")
def test_mse_loss_contract_expected_failure():
    torch.manual_seed(0)
    true_cost, batch = _make_standard_batch()
    pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)

    loss_fn = torch.nn.MSELoss(reduction="mean")
    _assert_loss_backward(loss_fn, pred, batch)


@pytest.mark.xfail(reason="CosineEmbeddingLoss expects input2/target, standardized batch does not include them yet.")
def test_cosine_embedding_loss_contract_expected_failure():
    torch.manual_seed(0)
    true_cost, batch = _make_standard_batch()
    pred = (true_cost + 0.05 * torch.randn_like(true_cost)).requires_grad_(True)

    loss_fn = torch.nn.CosineEmbeddingLoss(reduction="mean")
    _assert_loss_backward(loss_fn, pred, batch)
