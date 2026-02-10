from torch import nn
import torch
from torch.autograd import Function
from torch.utils.data import Dataset
import numpy as np

class PerturbedOpt(Function):
    """
    Autograd function for perturbation-based smoothing of an arbitrary loss.

    Contract:
      - Call via PerturbedOpt.apply(pred_cost, loss_eval, loss_kwargs, ...)
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
            return {k: PerturbedOpt.repeat_interleave_nested(v, s, dim=dim) for k, v in obj.items()}

        if isinstance(obj, tuple):
            return tuple(PerturbedOpt.repeat_interleave_nested(v, s, dim=dim) for v in obj)

        if isinstance(obj, list):
            return [PerturbedOpt.repeat_interleave_nested(v, s, dim=dim) for v in obj]

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

        # Base repetition factor
        # k = 2 * s if antithetic else s

        # Repeat each row
        X_rep = X.repeat_interleave(s, dim=0)  # (n*k, d)

        # Draw base noise
        noise = torch.randn(
            (n * s, d),
            device=X.device,
            dtype=X.dtype,
            generator=generator,
        )
        pos_perturbation = X_rep + sigma*noise

        if antithetic:
            neg_perturbation = X_rep - sigma*noise
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
            **kwargs):
        """
        Forward pass for perturbation-smoothed loss.

        Args:
            ctx: Context object to store information for backward computation
            pred_cost (torch.tensor): a batch of predicted values of the cost (B, d)
            loss_eval (callable): function called as loss_eval(pred_cost, **loss_kwargs)
            loss_kwargs (dict): keyword args passed to loss_eval; tensors inside will be
                repeat-interleaved to match perturbations (including nested dicts)
            sigma (float): noise scale for perturbations
            s (int): number of Monte Carlo samples per instance
            seed (int | None): RNG seed for perturbations
            antithetic (bool): whether to use antithetic sampling
            generator (torch.Generator | None): optional generator to use for noise
            control_variate (bool): whether to use a control variate baseline
            training (bool): whether to save tensors for backward pass
            
        Returns:
            torch.tensor: perturbation-smoothed loss per batch element
        """  

        batch_size = pred_cost.shape[0]

        # Sets up local random number generator and sets the seed (for reproduceability)
        gen = generator
        if gen is None:
            gen = torch.Generator(device=pred_cost.device)
            if seed is not None:
                gen.manual_seed(seed)

        # We now expand our datasets to estimate the expectation of the perturbed loss using Monte Carlo simulation with s samples
        # The functions expand the n samples in the batch into n*s samples. 
        noisy_pred_cost, neg_noisy_pred_cost, noise_out = PerturbedOpt.mc_expand_rows(pred_cost, sigma, s, generator=gen, antithetic=antithetic)

        expanded_loss_kwargs = PerturbedOpt.repeat_interleave_nested(loss_kwargs, s, dim=0)

        def _normalize_loss_shape(loss_out: torch.Tensor):
            if loss_out.ndim == 2 and loss_out.shape[1] == 1:
                return loss_out.squeeze(1)
            if loss_out.ndim >= 2 and loss_out.shape[-1] == 1:
                return loss_out.squeeze(-1)
            return loss_out
        
        if antithetic:
            # If antithetic we compute the loss for the negative of each draw and average it with the original draw
            with torch.no_grad():
                loss_out_pos = _normalize_loss_shape(loss_eval(noisy_pred_cost, **expanded_loss_kwargs))
                loss_out_neg = _normalize_loss_shape(loss_eval(neg_noisy_pred_cost, **expanded_loss_kwargs))
            loss_out = 0.5 * (loss_out_pos + loss_out_neg)
        else:
            # Computes the loss for each sample in the batch with the perturbation to the pred_cost vector (n * s total samples)
            with torch.no_grad():
                loss_out = _normalize_loss_shape(loss_eval(noisy_pred_cost, **expanded_loss_kwargs))
            loss_out_pos = loss_out
            loss_out_neg = None
        
        # Groups the Monte Carlo draws for each sample in the batch and computes the average loss 
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
            grad = (noise_out * (loss_pos - loss_neg) / (2 * sigma)).view(int(loss_pos.shape[0]/s), s, noise_out.shape[1]).mean(dim=1)
        else:
            grad = (loss_pos * noise_out).view(int(loss_pos.shape[0]/s), s, noise_out.shape[1]).mean(dim=1)/sigma

        if control_variate:
            cv = cv_loss
            if cv is not None and cv.ndim == 1:
                cv = cv.unsqueeze(1)
            grad = grad - noise_out.view(int(loss_pos.shape[0]/s), s, noise_out.shape[1]).mean(dim=1) * cv / sigma
        
        extra = (None,) * getattr(ctx, "extra_kwargs", 0)
        if grad_output.ndim == 1 and grad.ndim == 2:
            grad_output = grad_output.unsqueeze(1)
        return (grad_output * grad, None, None, None, None, None, None, None, None, None, *extra)
