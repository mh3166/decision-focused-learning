import torch
from torch import nn
from torch.autograd import Function


class RandomizedSmoothingFunc(Function):
    """
    Autograd function for perturbation-based smoothing of an arbitrary loss.

    Contract:
      - Call via RandomizedSmoothingFunc.apply(pred_cost, loss_eval, loss_kwargs, ...)
      - loss_eval: callable invoked as loss_eval(pred_cost, **loss_kwargs)
      - loss_kwargs: dict of keyword inputs; any tensors inside will be
        repeat-interleaved to match Monte Carlo perturbations (including
        nested dicts/tuples/lists).
      - loss_eval must return per-sample losses of shape (B*s,) or (B*s, 1).
    """

    @staticmethod
    def repeat_interleave_nested(obj, s: int, dim: int = 0):
        """
        Recursively repeat-interleave tensors inside nested structures.

        Supports tensors, dicts, tuples, and lists. Non-tensor leaves are
        returned unchanged.
        """
        def ri_if_tensor(x):
            return x.repeat_interleave(s, dim=dim) if isinstance(x, torch.Tensor) else x

        if isinstance(obj, torch.Tensor):
            return ri_if_tensor(obj)

        if isinstance(obj, dict):
            return {k: RandomizedSmoothingFunc.repeat_interleave_nested(v, s, dim=dim) for k, v in obj.items()}

        if isinstance(obj, tuple):
            return tuple(RandomizedSmoothingFunc.repeat_interleave_nested(v, s, dim=dim) for v in obj)

        if isinstance(obj, list):
            return [RandomizedSmoothingFunc.repeat_interleave_nested(v, s, dim=dim) for v in obj]

        return obj

    @staticmethod
    def mc_expand_rows(
        X: torch.Tensor,
        sigma: float,
        s: int,
        *,
        antithetic: bool = False,
        generator=None,
    ):
        """
        X: (n, d)
        Returns:
            pos_perturbation: (n * s, d)
            neg_perturbation: (n * s, d) (equals X_rep if antithetic=False)
            noise: (n * s, d)
        """
        n, d = X.shape

        X_rep = X.repeat_interleave(s, dim=0)

        noise = torch.randn(
            (n * s, d),
            device=X.device,
            dtype=X.dtype,
            generator=generator,
        )
        pos_perturbation = X_rep + sigma * noise

        if antithetic:
            neg_perturbation = X_rep - sigma * noise
        else:
            neg_perturbation = X_rep

        return pos_perturbation, neg_perturbation, noise

    @staticmethod
    def forward(
        ctx,
        pred_cost: torch.Tensor,
        loss_eval: callable,
        loss_kwargs: dict,
        sigma: float = 1,
        s: int = 1,
        seed: int | None = None,
        antithetic: bool = False,
        generator=None,
        control_variate: bool = False,
        training: bool = True,
        **kwargs,
    ):
        """
        Forward pass for perturbation-smoothed loss.
        """
        batch_size = pred_cost.shape[0]

        gen = generator
        if gen is None:
            gen = torch.Generator(device=pred_cost.device)
            if seed is not None:
                gen.manual_seed(seed)

        noisy_pred_cost, neg_noisy_pred_cost, noise_out = RandomizedSmoothingFunc.mc_expand_rows(
            pred_cost,
            sigma,
            s,
            generator=gen,
            antithetic=antithetic,
        )

        expanded_loss_kwargs = RandomizedSmoothingFunc.repeat_interleave_nested(loss_kwargs, s, dim=0)

        def _normalize_loss_shape(loss_out: torch.Tensor):
            if loss_out.ndim == 2 and loss_out.shape[1] == 1:
                return loss_out.squeeze(1)
            if loss_out.ndim >= 2 and loss_out.shape[-1] == 1:
                return loss_out.squeeze(-1)
            return loss_out

        if antithetic:
            with torch.no_grad():
                loss_out_pos = _normalize_loss_shape(loss_eval(noisy_pred_cost, **expanded_loss_kwargs))
                loss_out_neg = _normalize_loss_shape(loss_eval(neg_noisy_pred_cost, **expanded_loss_kwargs))
            loss_out = 0.5 * (loss_out_pos + loss_out_neg)
        else:
            with torch.no_grad():
                loss_out = _normalize_loss_shape(loss_eval(noisy_pred_cost, **expanded_loss_kwargs))
            loss_out_pos = loss_out
            loss_out_neg = None

        loss = loss_out.view(batch_size, s).mean(dim=1)

        ctx.extra_kwargs = len(kwargs)
        if training:
            if control_variate:
                with torch.no_grad():
                    cv_loss = _normalize_loss_shape(loss_eval(pred_cost, **loss_kwargs))
            else:
                cv_loss = None

            ctx.save_for_backward(loss_out_pos, loss_out_neg, noise_out, cv_loss)
            ctx.sigma = sigma
            ctx.s = s
            ctx.cv = control_variate
            ctx.anti = antithetic
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for perturbation-smoothed loss (grad w.r.t. pred_cost only).
        """
        loss_out_pos, loss_out_neg, noise_out, cv_loss = ctx.saved_tensors
        s = ctx.s
        sigma = ctx.sigma
        control_variate = ctx.cv
        antithetic = ctx.anti

        loss_pos = loss_out_pos
        loss_neg = loss_out_neg
        if loss_pos is not None and loss_pos.ndim == 1:
            loss_pos = loss_pos.unsqueeze(1)
        if loss_neg is not None and loss_neg.ndim == 1:
            loss_neg = loss_neg.unsqueeze(1)

        if antithetic:
            grad = (noise_out * (loss_pos - loss_neg) / (2 * sigma)).view(
                int(loss_pos.shape[0] / s),
                s,
                noise_out.shape[1],
            ).mean(dim=1)
        else:
            grad = (loss_pos * noise_out).view(
                int(loss_pos.shape[0] / s),
                s,
                noise_out.shape[1],
            ).mean(dim=1) / sigma

        if control_variate:
            cv = cv_loss
            if cv is not None and cv.ndim == 1:
                cv = cv.unsqueeze(1)
            grad = grad - noise_out.view(int(loss_pos.shape[0] / s), s, noise_out.shape[1]).mean(dim=1) * cv / sigma

        extra = (None,) * getattr(ctx, "extra_kwargs", 0)
        if grad_output.ndim == 1 and grad.ndim == 2:
            grad_output = grad_output.unsqueeze(1)
        return (grad_output * grad, None, None, None, None, None, None, None, None, None, *extra)


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

        out = RandomizedSmoothingFunc.apply(
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
