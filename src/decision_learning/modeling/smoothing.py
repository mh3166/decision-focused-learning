import torch
from torch import nn

from decision_learning.modeling.perturbed import PerturbedOpt


class RandomizedSmoothingWrapper(nn.Module):
    """
    Wrap a base loss module with perturbation-based randomized smoothing.

    The base loss must implement a standardized signature:
        per_sample(pred_cost, true_cost=None, true_sol=None, true_obj=None,
                  instance_kwargs=None, **kwargs) -> Tensor[B] or Tensor[B, 1]
        forward(...) -> reduced scalar or vector based on its reduction.
    """
    def __init__(
        self,
        base_loss: nn.Module,
        *,
        sigma: float,
        s: int,
        seed: int | None = None,
        antithetic: bool = False,
        control_variate: bool = False,
        reduction: str = "mean",
        training: bool = True,
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        if s < 1:
            raise ValueError("s must be >= 1")
        if sigma < 0:
            raise ValueError("sigma must be >= 0")

        self.base_loss = base_loss
        self.sigma = sigma
        self.s = s
        self.seed = seed
        self.antithetic = antithetic
        self.control_variate = control_variate
        self.reduction = reduction
        if not training:
            self.eval()

    def per_sample(
        self,
        pred_cost: torch.Tensor,
        true_cost: torch.Tensor | None = None,
        true_sol: torch.Tensor | None = None,
        true_obj: torch.Tensor | None = None,
        instance_kwargs: dict | None = None,
        **kwargs,
    ):
        # Pass through all standardized arguments and any extra kwargs so the
        # base loss can decide what it needs.
        loss_kwargs = {
            "true_cost": true_cost,
            "true_sol": true_sol,
            "true_obj": true_obj,
            "instance_kwargs": instance_kwargs,
            **kwargs,
        }

        def loss_eval(noisy_pred, **kw):
            # Use the base loss per-sample path for perturbation smoothing.
            return self.base_loss.per_sample(pred_cost=noisy_pred, **kw)

        out = PerturbedOpt.apply(
            pred_cost,
            loss_eval,
            loss_kwargs,
            self.sigma,
            self.s,
            self.seed,
            self.antithetic,
            None,
            self.control_variate,
            self.training,
        )

        # Normalize to a flat per-sample vector.
        if out.ndim == 2 and out.shape[1] == 1:
            out = out.squeeze(1)
        return out

    def forward(
        self,
        pred_cost: torch.Tensor,
        true_cost: torch.Tensor | None = None,
        true_sol: torch.Tensor | None = None,
        true_obj: torch.Tensor | None = None,
        instance_kwargs: dict | None = None,
        **kwargs,
    ):
        # Mirror standard loss module behavior: per-sample then reduction.
        loss = self.per_sample(
            pred_cost,
            true_cost=true_cost,
            true_sol=true_sol,
            true_obj=true_obj,
            instance_kwargs=instance_kwargs,
            **kwargs,
        )
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()
