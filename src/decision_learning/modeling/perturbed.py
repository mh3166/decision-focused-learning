from torch import nn
import torch
from torch.autograd import Function
from torch.utils.data import Dataset
import numpy as np

class PerturbedOpt(Function):
    """
    A autograd function for Perturbation Smoothing
    """
	
    @staticmethod
    def repeat_interleave_nested(obj, s: int, dim: int = 0):
        """
        Apply repeat_interleave to:
          - tensors directly
          - elements of a tuple if they are tensors
          - tensor VALUES of dicts that appear as elements of a tuple

        Does NOT recurse beyond:
          tuple -> dict (one level), i.e., no recursion inside nested dicts/tuples.
        """
        def ri_if_tensor(x):
            return x.repeat_interleave(s, dim=dim) if isinstance(x, torch.Tensor) else x

        # If it's a tensor: apply
        if isinstance(obj, torch.Tensor):
            return ri_if_tensor(obj)

        # If it's a dict: apply ONLY to its immediate tensor values (no recursion)
        if isinstance(obj, dict):
            return {k: ri_if_tensor(v) for k, v in obj.items()}

        # If it's a tuple: apply to immediate elements; dict elements get one-level handling
        if isinstance(obj, tuple):
            out = []
            for v in obj:
                if isinstance(v, torch.Tensor):
                    out.append(ri_if_tensor(v))
                elif isinstance(v, dict):
                    out.append({k: ri_if_tensor(val) for k, val in v.items()})
                else:
                    out.append(v)
            return tuple(out)

        # Anything else unchanged
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
            X_mc: (n * s, d) or (n * 2s, d) if antithetic=True
            noise: same shape as X_mc
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
    def forward(ctx, 
            pred_cost: torch.tensor, 
            loss_args: tuple,
            loss_fn: callable,
           	sigma: float = 1,
           	s: int = 1,
            antithetic: bool = False,
            control_variate: bool = False,
            seed: int = 42,
            training: bool = True):
        """
        Forward pass for SPO+

        Args:
            ctx: Context object to store information for backward computation
            pred_cost (torch.tensor): a batch of predicted values of the cost            
            optmodel (callable): a function/class that solves an optimization problem using pred_cost. For every batch of data, we use
                optmodel to solve the optimization problem using the predicted cost to get the optimal solution and objective value.
                It must take in:                                 
                    - pred_cost (torch.tensor): predicted coefficients/parameters for optimization model
                    - instance_kwargs (dict): a dictionary of per-sample arrays of data that define each optimization instance
                It must also:
                    - detach tensors if necessary
                    - loop or batch data solve 
                In practice, the user should wrap their own optmodel in the decision_learning.utils.handle_solver function so that
                these are all taken care of.                                
            minimize (bool): whether the optimization problem is minimization or maximization            
            instance_kwargs (dict): a dictionary of per-sample arrays of data that define each optimization instance
            
        Returns:
            torch.tensor: SPO+ loss
        """  

        batch_size = pred_cost.shape[0]

        # Sets up local random number generator and sets the seed (for reproduceability) 
        gen = torch.Generator(device=pred_cost.device)
        gen.manual_seed(seed)

        # We now expand our datasets to estimate the expectation of the perturbed loss using Monte Carlo simulation with s samples
        # The functions expand the n samples in the batch into n*s samples. 
        noisy_pred_cost, neg_noisy_pred_cost, noise_out = PerturbedOpt.mc_expand_rows(pred_cost, sigma, s, generator=gen, antithetic=antithetic)

        expanded_loss_args = PerturbedOpt.repeat_interleave_nested(loss_args, s)
        
        if antithetic:
            # If antithetic we compute the loss for the negative of each draw and average it with the original draw
            loss_out_pos = loss_fn.apply(noisy_pred_cost, *expanded_loss_args)
            loss_out_neg = loss_fn.apply(neg_noisy_pred_cost, *expanded_loss_args)
            loss_out = 0.5*(loss_out_pos + loss_out_neg)
        else:
            # Computes the loss for each sample in the batch with the perturbation to the pred_cost vector (n * s total samples)
            loss_out = loss_fn.apply(noisy_pred_cost, *expanded_loss_args)
            loss_out_pos = loss_out
            loss_out_neg = None
        
        # Groups the Monte Carlo draws for each sample in the batch and computes the average loss 
        loss = loss_out.view(batch_size, s, loss_out.shape[1]).mean(dim = 1)

        if training:
            if control_variate:
                cv_loss = loss_fn.apply(pred_cost, *loss_args)
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
        Backward pass for Perturbation Smoothed Function
        """
        loss_out_pos, loss_out_neg, noise_out, cv_loss = ctx.saved_tensors
        s = ctx.s
        sigma = ctx.sigma
        control_variate = ctx.cv
        antithetic = ctx.anti

        if antithetic:
            grad = (noise_out * (loss_out_pos - loss_out_neg) / (2 * sigma)).view(int(loss_out_pos.shape[0]/s), s, noise_out.shape[1]).mean(dim=1)
        else:
            grad = (loss_out_pos * noise_out).view(int(loss_out_pos.shape[0]/s), s, noise_out.shape[1]).mean(dim=1)/sigma

        if control_variate:
            grad = grad - noise_out.view(int(loss_out_pos.shape[0]/s), s, noise_out.shape[1]).mean(dim=1) * cv_loss / sigma
        
        return grad_output * grad, None, None, None, None, None, None, None
