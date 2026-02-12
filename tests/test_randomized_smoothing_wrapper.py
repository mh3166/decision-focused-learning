import torch

from decision_learning.modeling.smoothing import RandomizedSmoothingWrapper
from decision_learning.modeling.loss import PGLoss


class ToyLoss(torch.nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def per_sample(self, pred_cost, true_cost=None, **kw):
        return (pred_cost * true_cost).sum(dim=1)

    def forward(self, pred_cost, true_cost=None, **kw):
        ps = self.per_sample(pred_cost, true_cost=true_cost, **kw)
        return ps.mean()


def test_randomized_smoothing_wrapper_contract_and_backward_toy_loss():
    torch.manual_seed(0)
    B, d = 5, 4
    pred = torch.randn(B, d, requires_grad=True)
    true_cost = torch.randn(B, d)

    base = ToyLoss()
    wrapper = RandomizedSmoothingWrapper(
        base_loss=base,
        sigma=0.2,
        s=8,
        seed=0,
        antithetic=False,
        control_variate=False,
        reduction="mean",
    )

    loss = wrapper(
        pred_cost=pred,
        true_cost=true_cost,
        true_sol=None,
        true_obj=None,
        instance_kwargs=None,
    )

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()

    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == (B, d)
    assert torch.isfinite(pred.grad).all()


def test_randomized_smoothing_wrapper_uses_per_sample_reduction_none():
    torch.manual_seed(0)
    B, d = 4, 3
    pred = torch.randn(B, d, requires_grad=True)
    true_cost = torch.randn(B, d)

    base = ToyLoss()
    wrapper = RandomizedSmoothingWrapper(
        base_loss=base,
        sigma=0.1,
        s=4,
        seed=123,
        antithetic=False,
        control_variate=False,
        reduction="none",
    )

    out = wrapper(
        pred_cost=pred,
        true_cost=true_cost,
        true_sol=None,
        true_obj=None,
        instance_kwargs=None,
    )

    assert out.shape == (B,)
    assert torch.isfinite(out).all()


def _box_oracle_torch(costs: torch.Tensor, b: torch.Tensor):
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


def test_randomized_smoothing_wrapper_with_pg_loss():
    torch.manual_seed(0)
    B, d = 4, 3
    pred = torch.randn(B, d, requires_grad=True)
    true_cost = torch.tensor(
        [[-1.0, 2.0, -0.5], [0.5, -1.0, 1.5], [-2.0, 0.1, 0.2], [1.0, -0.2, -0.3]],
        dtype=torch.float32,
    )
    instance_kwargs = {"b": torch.ones_like(true_cost)}

    base = PGLoss(
        optmodel=_box_oracle_with_kwargs,
        h=0.1,
        finite_diff_type="B",
        reduction="none",
        minimize=True,
    )
    wrapper = RandomizedSmoothingWrapper(
        base_loss=base,
        sigma=0.1,
        s=4,
        seed=0,
        antithetic=False,
        control_variate=False,
        reduction="mean",
    )

    loss = wrapper(
        pred_cost=pred,
        true_cost=true_cost,
        true_sol=None,
        true_obj=None,
        instance_kwargs=instance_kwargs,
    )

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()

    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == (B, d)
    assert torch.isfinite(pred.grad).all()


def test_randomized_smoothing_wrapper_antithetic_path():
    torch.manual_seed(0)
    B, d = 4, 3
    pred = torch.randn(B, d, requires_grad=True)
    true_cost = torch.randn(B, d)

    base = ToyLoss()
    wrapper = RandomizedSmoothingWrapper(
        base_loss=base,
        sigma=0.2,
        s=6,
        seed=42,
        antithetic=True,
        control_variate=False,
        reduction="mean",
    )

    loss = wrapper(pred_cost=pred, true_cost=true_cost)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()

    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == (B, d)
    assert torch.isfinite(pred.grad).all()


def test_randomized_smoothing_wrapper_control_variate_path():
    torch.manual_seed(0)
    B, d = 4, 3
    pred = torch.randn(B, d, requires_grad=True)
    true_cost = torch.randn(B, d)

    base = ToyLoss()
    wrapper = RandomizedSmoothingWrapper(
        base_loss=base,
        sigma=0.1,
        s=5,
        seed=7,
        antithetic=False,
        control_variate=True,
        reduction="mean",
    )

    loss = wrapper(pred_cost=pred, true_cost=true_cost)
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()

    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == (B, d)
    assert torch.isfinite(pred.grad).all()


def test_randomized_smoothing_wrapper_accepts_column_loss():
    torch.manual_seed(0)
    B, d = 4, 3
    pred = torch.randn(B, d, requires_grad=True)
    true_cost = torch.randn(B, d)

    class ColumnToyLoss(torch.nn.Module):
        def per_sample(self, pred_cost, true_cost=None, **kw):
            return (pred_cost * true_cost).sum(dim=1, keepdim=True)

        def forward(self, pred_cost, true_cost=None, **kw):
            return self.per_sample(pred_cost, true_cost=true_cost, **kw).mean()

    base = ColumnToyLoss()
    wrapper = RandomizedSmoothingWrapper(
        base_loss=base,
        sigma=0.15,
        s=4,
        seed=3,
        antithetic=False,
        control_variate=False,
        reduction="none",
    )

    out = wrapper(pred_cost=pred, true_cost=true_cost)
    assert out.shape == (B,)
    assert torch.isfinite(out).all()
