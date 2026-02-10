import torch

from decision_learning.modeling.perturbed import PerturbedOpt


def test_perturbed_forward_uses_no_grad_for_loss_eval():
    torch.manual_seed(0)
    B, d = 4, 3

    pred_cost = torch.randn(B, d, requires_grad=True)
    y = torch.randn(B, d)

    def loss_eval(pred, **kw):
        target = kw["y"]
        return ((pred - target) ** 2).sum(dim=1)

    loss = PerturbedOpt.apply(
        pred_cost,
        loss_eval,
        {"y": y},
        0.1,
        8,
        0,
    )

    assert loss.shape == (B,)
    assert loss.grad_fn is not None
    assert pred_cost.grad is None

    loss.mean().backward()

    assert pred_cost.grad is not None
    assert pred_cost.grad.shape == (B, d)
    assert torch.isfinite(pred_cost.grad).all()
