from torch import nn
import torch
import copy
from torch.autograd import Function
from torch.utils.data import Dataset
import numpy as np
from decision_learning.utils import handle_solver


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return torch.mean(loss)
    if reduction == "sum":
        return torch.sum(loss)
    if reduction == "none":
        return loss
    raise ValueError("No reduction '{}'.".format(reduction))


def _normalize_per_sample(loss: torch.Tensor) -> torch.Tensor:
    if loss.ndim == 2 and loss.shape[1] == 1:
        return loss.squeeze(1)
    return loss


def CILO_lbda(
    pred_cost: torch.Tensor,
    Y: torch.Tensor,
    optmodel,
    beta: float,
    *,
    kwargs: dict | None = None,
    sens: float = 1e-4,
    init_upper: float = 1e-2,
    max_expand: int = 50,
    max_bisect: int = 200,
) -> float:
    """
    Find lambda >= 0 such that mean_i <Y_i, x_i(lambda)> - beta ~= 0,
    where x_i(lambda) is returned by optmodel(pred_cost + lambda * Y).

    Uses:
      1) Exponential search to bracket a sign change
      2) Bisection to solve to tolerance

    Note: res_u and res_l are not necessary, but may be useful for debugging
    """
    if kwargs is None:
        kwargs = {}

    def residual(lbda: float) -> float:
        sol, _ = optmodel(pred_cost + lbda * Y, **kwargs)  # sol: (batch, d)
        # mean over batch of dot(Y_i, sol_i)
        return (torch.sum(Y * sol, dim=1).mean() - beta).item()

    # ---- 1) Bracket root: find [lbda_l, lbda_u] with res(l) > 0 and res(u) <= 0 ----
    lbda_l = 0.0
    lbda_u = float(init_upper)

    res_u = residual(lbda_u)
    expand_steps = 0
    while res_u > 0 and expand_steps < max_expand:
        lbda_l = lbda_u
        lbda_u *= 2.0
        res_u = residual(lbda_u)
        expand_steps += 1

    # If we never found res_u <= 0, we can't bracket; return best effort upper.
    if res_u > 0:
        return lbda_u

    # ---- 2) Bisection on bracket ----
    res_l = residual(lbda_l)  # should typically be > 0 for a proper bracket
    lbda = 0.5 * (lbda_l + lbda_u)

    for _ in range(max_bisect):
        res_mid = residual(lbda)

        if abs(res_mid) <= sens:
            break

        # Maintain invariant: res(l) > 0, res(u) <= 0
        if res_mid > 0:
            lbda_l, res_l = lbda, res_mid
        else:
            lbda_u, res_u = lbda, res_mid

        new_lbda = 0.5 * (lbda_l + lbda_u)

        # Optional early stop: interval too small and we're on the feasible side
        if abs(new_lbda - lbda) < sens and res_mid <= 0:
            lbda = new_lbda
            break

        lbda = new_lbda

    return float(lbda)

# -------------------------------------------------------------------------
# SPO Plus (Smart Predict and Optimize Plus) Loss
# -------------------------------------------------------------------------

class SPOPlusLoss(nn.Module):
    """
    Wrapper function around SPOLossFunc with customized backwards pass.
    This loss uses manual backward logic (not vanilla autograd). Extend
    from nn.Module to use nn.Module's functionalities.
    
    The SPO+ Loss is convex with subgradient. Thus, it allows us to design an
    algorithm based on stochastic gradient descent.

    Reference: <https://doi.org/10.1287/mnsc.2020.3922>
    """

    def __init__(self, 
                optmodel: callable, 
                is_minimization: bool=True,
                reduction: str="mean"):
        """
        Args:
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
            reduction (str): the reduction to apply to the output
            is_minimization (bool): whether the optimization problem is minimization or maximization              
        """
        super(SPOPlusLoss, self).__init__()        
        self.reduction = reduction
        self.is_minimization = is_minimization
        self.optmodel = optmodel      



    def forward(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        loss = self.per_sample(
            pred_cost,
            obs_cost=obs_cost,
            obs_sol=obs_sol,
            obs_obj=obs_obj,
            instance_kwargs=instance_kwargs,
            **kwargs,
        )
        return _reduce_loss(loss, self.reduction)

    def per_sample(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        """
        Per-sample loss.
        
        Args:            
            pred_cost (torch.tensor): a batch of predicted values of the cost            
            obs_cost (torch.tensor): a batch of observed cost values
            obs_sol (torch.tensor): a batch of solutions w.r.t. obs_cost
            obs_obj (torch.tensor): a batch of objectives w.r.t. obs_cost and obs_sol            

        """
        if instance_kwargs is None:
            instance_kwargs = {}
        loss = SPOPlusLossFunc.apply(pred_cost, 
                            obs_cost, 
                            obs_sol, 
                            obs_obj, 
                            self.optmodel, 
                            self.is_minimization,                             
                            instance_kwargs
                        )
        return _normalize_per_sample(loss)
    
    
class SPOPlusLossFunc(Function):
    """
    A autograd function for SPO+ Loss with a custom gradient (manual backward).
    """

    @staticmethod
    def forward(ctx, 
            pred_cost: torch.tensor, 
            obs_cost: torch.tensor, 
            obs_sol: torch.tensor, 
            obs_obj: torch.tensor,
            optmodel: callable,
            is_minimization: bool = True,            
            instance_kwargs: dict = {}):
        """
        Forward pass for SPO+.

        Args:
            ctx: Context object to store information for backward computation
            pred_cost (torch.tensor): a batch of predicted values of the cost            
            obs_cost (torch.tensor): a batch of observed cost values
            obs_sol (torch.tensor): a batch of solutions w.r.t. obs_cost
            obs_obj (torch.tensor): a batch of objectives w.r.t. obs_cost and obs_sol
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
            is_minimization (bool): whether the optimization problem is minimization or maximization            
            instance_kwargs (dict): a dictionary of per-sample arrays of data that define each optimization instance
            
        Returns:
            torch.tensor: SPO+ loss
        """        
        # rename variable names for convenience
        # c for cost, w for solution variables, z for obj values, and we use _hat for variables derived from predicted values
        c_hat = pred_cost
        c, w, z = obs_cost, obs_sol, obs_obj
        
        # get batch's current optimal solution value and objective vvalue based on the predicted cost
        w_hat, z_hat = optmodel(2*c_hat - c, **instance_kwargs)                            
                        
        # calculate loss
        # SPO loss = - min_{w} (2 * c_hat - c)^T w + 2 * c_hat^T w - z = - z_hat + 2 * c_hat^T w - z
        loss = - z_hat + 2 * torch.sum(c_hat * w, axis = 1).reshape(-1,1) - z
        if not is_minimization:
            loss = - loss
        
        # save solutions for backwards pass
        ctx.save_for_backward(w, w_hat)
        ctx.is_minimization = is_minimization
        
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        w, w_hat = ctx.saved_tensors
  
        if ctx.is_minimization:
            grad = 2 * (w - w_hat)
        else:
            grad = 2 * (w_hat - w)
       
        return grad_output * grad, None, None, None, None, None, None, None


# -------------------------------------------------------------------------
# Perturbation Gradient (PG) Loss
# -------------------------------------------------------------------------

class PGLoss(nn.Module):
    """
    An autograd module for Perturbation Gradient (PG) Loss using a custom gradient
    (manual backward, not vanilla autograd).

    Reference: <https://arxiv.org/pdf/2402.03256>
    """

    def __init__(self, 
                optmodel: callable, 
                h: float=1, 
                finite_diff_type: str='B', 
                reduction: str="mean", 
                is_minimization: bool=True,
                scale_by_norm: bool=False,
            ):                 
        """
        Args:
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
            h (float): perturbation size/finite difference step size for zeroth order gradient approximation
            finite_diff_type (str, optional): Specify type of finite-difference scheme:
                                            - Backward Differencing/PGB ('B')
                                            - Central Differencing/PGC ('C')
                                            - Forward Differencing/PGF ('F')
            reduction (str): the reduction to apply to the output
            is_minimization (bool): whether the optimization problem is minimization or maximization
            scale_by_norm (bool): whether to scale per-sample perturbations by l2 norm
        """
        # the finite difference step size h must be positive
        if h < 0:
            raise ValueError("h must be positive")
        # finite difference scheme must be one of the following
        if finite_diff_type not in ['B', 'C', 'F']:
            raise ValueError("finite_diff_type must be one of 'B', 'C', 'F'")
        
        super(PGLoss, self).__init__()     
        self.h = h
        self.finite_diff_type = finite_diff_type
        self.reduction = reduction
        self.is_minimization = is_minimization
        self.optmodel = optmodel
        self.scale_by_norm = scale_by_norm
        

    def forward(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        loss = self.per_sample(
            pred_cost,
            obs_cost=obs_cost,
            obs_sol=obs_sol,
            obs_obj=obs_obj,
            instance_kwargs=instance_kwargs,
            **kwargs,
        )
        return _reduce_loss(loss, self.reduction)

    def per_sample(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        """
        Per-sample loss.
        
        Args:            
            pred_cost (torch.tensor): a batch of predicted values of the cost            
            obs_cost (torch.tensor): a batch of observed cost values
        """
        if instance_kwargs is None:
            instance_kwargs = {}
        loss = PGFunc.apply(pred_cost, 
                            obs_cost, 
                            self.h, 
                            self.finite_diff_type, 
                            self.optmodel,
                            self.is_minimization,
                            self.scale_by_norm,                                           
                            instance_kwargs
                        )
        return _normalize_per_sample(loss)
    
    
class PGFunc(Function):
    """
    A autograd function for Perturbation Gradient (PG) Loss with a custom gradient
    (manual backward).
    """

    @staticmethod
    def forward(ctx, 
            pred_cost: torch.tensor, 
            obs_cost: torch.tensor, 
            h: float, 
            finite_diff_type: str,
            optmodel: callable,
            is_minimization: bool = True,
            scale_by_norm: bool = False,
            instance_kwargs: dict = {}):            
        """
        Forward pass for PG Loss.

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            obs_cost (torch.tensor): a batch of observed cost values
            h (float): perturbation size/finite difference step size for zeroth order gradient approximation
            finite_diff_type (str, optional): Specify type of finite-difference scheme:
                                            - Backward Differencing/PGB ('B')
                                            - Central Differencing/PGC ('C')
                                            - Forward Differencing/PGF ('F')
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
            is_minimization (bool): whether the optimization problem is minimization or maximization         
            scale_by_norm (bool): whether to scale per-sample perturbations by l2 norm
            instance_kwargs (dict): a dictionary of per-sample arrays of data that define each optimization instance
            
            
        Returns:
            torch.tensor: PG loss
        """        
        # detach (stops gradient tracking since we will compute custom gradient) and move to cpu. Do this since
        # generally the optmodel is probably cpu based
        cp = pred_cost 
        c = obs_cost 

        if scale_by_norm:
            norms = torch.norm(c, dim=1, keepdim=True)
            scaled_c = torch.where(norms > 0, c / norms, c)
        else:
            scaled_c = c

        # for PG loss with zeroth order gradients, we need to perturb the predicted costs and solve
        # two optimization problems to approximate the gradient, where there is a cost plus and minus perturbation
        # that changes depending on the finite difference scheme.
        if finite_diff_type == 'C': # central diff: (1/2h) * (optmodel(pred_cost + h*obs_cost) - optmodel(pred_cost - h*obs_cost))
            cp_plus = cp + h * scaled_c
            cp_minus = cp - h * scaled_c
            step_size = 1 / (2 * h)
        elif finite_diff_type == 'B': # back diff: (1/h) * (optmodel(pred_cost) - optmodel(pred_cost - h*obs_cost))
            cp_plus = cp
            cp_minus = cp - h * scaled_c
            step_size = 1 / h
        elif finite_diff_type == 'F': # forward diff: (1/h) * (optmodel(pred_cost + h*obs_cost) - optmodel(pred_cost))
            cp_plus = cp + h * scaled_c
            cp_minus = cp
            step_size = 1 / h

        # solve optimization problems
        # Plus Perturbation Optimization Problem
        sol_plus, obj_plus = optmodel(cp_plus, **instance_kwargs)

        # Minus Perturbation Optimization Problem
        sol_minus, obj_minus = optmodel(cp_minus, **instance_kwargs)   
        
        # calculate loss
        loss = (obj_plus - obj_minus) * step_size
        if not is_minimization:
            loss = - loss
                
        # save solutions and objects needed for backwards pass to compute gradients
        ctx.save_for_backward(sol_plus, sol_minus)        
        ctx.is_minimization = is_minimization
        ctx.step_size = step_size
        return loss


    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for PG Loss
        """
        sol_plus, sol_minus = ctx.saved_tensors  
        step_size = ctx.step_size

        # below, need to move (sol_plus - sol_minus) to the same device as grad_output since sol_plus and sol_minus
        # are on cpu and it is possible that grad_output is on a different device
        grad = step_size * (sol_plus - sol_minus).to(grad_output.device)
        if not ctx.is_minimization: # maximization problem case
            grad = - grad
        
        return grad_output * grad, None, None, None, None, None, None, None


# -------------------------------------------------------------------------
# Fenchel-Young (unsmoothed) Loss
# -------------------------------------------------------------------------

class FYLoss(nn.Module):
    """
    Autograd module for the (unsmoothed) Fenchel-Young loss using a custom gradient.
    """

    def __init__(
        self,
        optmodel: callable,
        reduction: str = "mean",
        is_minimization: bool = True,
    ):
        """
        Args:
            optmodel (callable): optimization model called as optmodel(cost, **instance_kwargs)
                returning (solution, objective)
            reduction (str): the reduction to apply to the output
            is_minimization (bool): whether the optimization problem is minimization or maximization
        """
        super().__init__()
        self.optmodel = optmodel
        self.reduction = reduction
        self.is_minimization = is_minimization

    def forward(
        self,
        pred_cost: torch.Tensor,
        obs_cost: torch.Tensor | None = None,
        obs_sol: torch.Tensor | None = None,
        obs_obj: torch.Tensor | None = None,
        instance_kwargs: dict | None = None,
        **kwargs,
    ):
        loss = self.per_sample(
            pred_cost,
            obs_cost=obs_cost,
            obs_sol=obs_sol,
            obs_obj=obs_obj,
            instance_kwargs=instance_kwargs,
            **kwargs,
        )
        return _reduce_loss(loss, self.reduction)

    def per_sample(
        self,
        pred_cost: torch.Tensor,
        obs_cost: torch.Tensor | None = None,
        obs_sol: torch.Tensor | None = None,
        obs_obj: torch.Tensor | None = None,
        instance_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Per-sample loss.
        """
        if instance_kwargs is None:
            instance_kwargs = {}
        loss = FYFunc.apply(
            pred_cost,
            obs_sol,
            self.optmodel,
            self.is_minimization,
            instance_kwargs,
        )
        return _normalize_per_sample(loss)


class FYFunc(Function):
    """
    Autograd function for the (unsmoothed) Fenchel-Young loss with custom gradient.
    """

    @staticmethod
    def forward(ctx,
            pred_cost: torch.tensor,
            obs_sol: torch.tensor,
            optmodel: callable,
            is_minimization: bool = True,
            instance_kwargs: dict = {}):
        """
        Forward pass for unsmoothed Fenchel-Young loss.

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            obs_sol (torch.tensor): a batch of solutions w.r.t. obs_cost
            optmodel (callable): solves the optimization problem given pred_cost
            is_minimization (bool): whether the optimization problem is minimization or maximization
            instance_kwargs (dict): per-sample data defining each optimization instance

        Returns:
            torch.tensor: per-sample Fenchel-Young loss
        """
        # Notation: T is predicted costs, z_star is true solution, z_T is predicted solution.
        # Let optmodel handle any detaching / device moves as needed.
        T = pred_cost
        z_star = obs_sol

        with torch.no_grad():
            z_T, _ = optmodel(T, **instance_kwargs)

        # Inner product <pred_cost, pred_sol - obs_sol>
        loss = torch.sum(T * (z_star - z_T), dim=1)
        if not is_minimization:
            loss = -loss

        # Save tensors for backward (kept on cpu).
        ctx.save_for_backward(z_T, z_star)
        ctx.is_minimization = is_minimization

        return loss.to(device=pred_cost.device, dtype=pred_cost.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for Fenchel-Young loss.
        """
        z_T, z_star = ctx.saved_tensors
        grad = (z_T - z_star).to(grad_output.device)
        if not ctx.is_minimization:
            grad = -grad

        if grad_output.ndim == 1:
            grad_output = grad_output.unsqueeze(1)
        return grad_output * grad, None, None, None, None
    

# -------------------------------------------------------------------------
# Adaptive PG Loss
# -------------------------------------------------------------------------

class PGAdaptiveLoss(nn.Module):
    """
    Adaptive PG loss where the perturbation scale h is chosen via CILO_lbda
    based on the current batch. pred_cost is computed externally and passed in.
    """
    def __init__(self, optmodel, 
                    beta: float,
                    reduction: str="mean", 
                    is_minimization: bool=True):
        super().__init__()
        self.optmodel = optmodel
        self.beta = beta
        self.reduction = reduction
        self.is_minimization = is_minimization

    def forward(
        self,
        pred_cost: torch.Tensor,   # f_theta(w) computed outside
        obs_cost: torch.Tensor | None = None,
        obs_sol: torch.Tensor | None = None,
        obs_obj: torch.Tensor | None = None,
        instance_kwargs: dict | None = None,  # per-sample data defining the optimization instance (e.g., feasible region).
        **kwargs,
    ):
        loss = self.per_sample(
            pred_cost,
            obs_cost=obs_cost,
            obs_sol=obs_sol,
            obs_obj=obs_obj,
            instance_kwargs=instance_kwargs,
            **kwargs,
        )
        return _reduce_loss(loss, self.reduction)

    def per_sample(
        self,
        pred_cost: torch.Tensor,   # f_theta(w) computed outside
        obs_cost: torch.Tensor | None = None,
        obs_sol: torch.Tensor | None = None,
        obs_obj: torch.Tensor | None = None,
        instance_kwargs: dict | None = None,  # per-sample data defining the optimization instance (e.g., feasible region).
        **kwargs,
    ):
        t = pred_cost
        y = obs_cost
        if instance_kwargs is None:
            instance_kwargs = {}

        with torch.no_grad():
            h = CILO_lbda(t, y, self.optmodel, self.beta, kwargs=instance_kwargs)

        t_plus = t + h * y

        # ----------------------------------------------------
        # Compute objective terms at t and t + h y
        # ----------------------------------------------------
        with torch.no_grad():
            x_t, obj_t = self.optmodel(t, **instance_kwargs)
            x_t, obj_t = x_t.detach(), obj_t.detach()
            x_t_plus, obj_t_plus = self.optmodel(t_plus, **instance_kwargs)
            x_t_plus, obj_t_plus = x_t_plus.detach(), obj_t_plus.detach()

        term1 = torch.sum(t_plus * x_t_plus, axis = 1)
        # print("term 1: ", term1.mean().item(), torch.sum(x_t_plus, axis = 1).mean().item())

        # Live term: gradients flow through t and through x_fn(t-hc) if x_fn supports autograd
        term2 = torch.sum(t * x_t, axis = 1)
        # print("term 2: ", term2.mean().item(), torch.sum(x_t_plus, axis = 1).mean().item())

        loss = (term1 - term2) / h
        return _normalize_per_sample(loss)

# -------------------------------------------------------------------------
# Cosine Surrogates
# -------------------------------------------------------------------------
class CosineSurrogateDotProdMSELoss(nn.Module):
    """Implements a convexified surrogate loss function for cosine similarity loss by taking
    a linear combination of the mean squared error (MSE) and the dot product of the predicted and true costs since
    - MSE captures magnitude of the difference between predicted and true costs
    - Dot product captures the direction/angle of the difference between predicted and true costs
    """
    
    def __init__(self, alpha: float=1, reduction: str='mean', is_minimization: bool=True):
        """
        Args:
            alpha (float, optional): Weighting parameter for how heavily to weigh MSE component of loss vs dot product. Defaults to 1.
            reduction (str): the reduction to apply to the output. Defaults to 'mean'.
            is_minimization (bool): whether the optimization problem is minimization or maximization. Defaults to True.
        """
        super(CosineSurrogateDotProdMSELoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.is_minimization = is_minimization
        self.mse_loss = nn.MSELoss(reduction="none") # use off-the-shelf MSE loss
        

    def forward(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        loss = self.per_sample(
            pred_cost,
            obs_cost=obs_cost,
            obs_sol=obs_sol,
            obs_obj=obs_obj,
            instance_kwargs=instance_kwargs,
            **kwargs,
        )
        return _reduce_loss(loss, self.reduction)

    def per_sample(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        """Takes the predicted and true costs and computes the loss using the convexified cosine surrogate loss function
        that is linear combination of MSE and dot product of predicted and true costs.
        
        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            obs_cost (torch.tensor): a batch of observed cost values        
        """        
        mse = self.mse_loss(pred_cost, obs_cost)
        if mse.ndim > 1:
            mse = mse.view(mse.shape[0], -1).mean(dim=1)
        
        # ----- Compute dot product -----
        dot_product = torch.sum(pred_cost * obs_cost, dim=1)
        if self.is_minimization:
            dot_product = -dot_product # negate dot product for minimization
        
        loss = self.alpha * mse + dot_product  # compute final loss as linear combination of MSE and dot product        
       
        return _normalize_per_sample(loss)
    
    
class CosineSurrogateDotProdVecMagLoss(nn.Module):
    """Implements a convexified surrogate loss function for cosine similarity loss by taking
    trying to maximize the dot product of the predicted and true costs while simultaneously minimizing the magnitude of the predicted cost
    since this would incentivize the predicted cost to be in the same direction as the true cost without the predictions artificially
    making the dot product higher by increasing the magnitude of the predicted cost.    
    """
    def __init__(self, alpha: float=1, reduction: str='mean', is_minimization: bool=True):
        """
        Args:
            alpha (float, optional): Weight emphasis on minimizing magnitude of predicted vector (measured through self dot product). Defaults to 1.
            reduction (str): the reduction to apply to the output. Defaults to 'mean'.
            is_minimization (bool): whether the optimization problem is minimization or maximization. Defaults to True.
        """
        super(CosineSurrogateDotProdVecMagLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.is_minimization = is_minimization
        

    def forward(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        loss = self.per_sample(
            pred_cost,
            obs_cost=obs_cost,
            obs_sol=obs_sol,
            obs_obj=obs_obj,
            instance_kwargs=instance_kwargs,
            **kwargs,
        )
        return _reduce_loss(loss, self.reduction)

    def per_sample(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        """Computes the loss using a linear combination of two components:
        1) self dot product - measures the magnitude of the predicted cost vector, trying to is_minimization it
        2) dot product of predicted and true costs - measures the direction of the predicted cost vector, trying to maximize it

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            obs_cost (torch.tensor): a batch of observed cost values          
        """    
        dot_product_self = torch.sum(pred_cost * pred_cost, dim=1)
        
        # dot product of predicted and true costs - measure angle between predicted and true costs
        dot_product_ang = torch.sum(pred_cost * obs_cost, dim=1)
        if self.is_minimization:
            dot_product_ang = -dot_product_ang # negate dot product for minimization
        
        loss = self.alpha * dot_product_self + dot_product_ang  # compute final loss as linear combination of self dot product and dot product        
       
        return _normalize_per_sample(loss)


# -------------------------------------------------------------------------
# Standardized wrappers for PyTorch losses
# -------------------------------------------------------------------------

class MSELoss(nn.Module):
    """Wrapper around nn.MSELoss with standardized loss signature."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.reduction = reduction

    def forward(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        if obs_cost is None:
            raise ValueError("StandardMSELoss requires obs_cost.")
        loss = self.per_sample(
            pred_cost,
            obs_cost=obs_cost,
            obs_sol=obs_sol,
            obs_obj=obs_obj,
            instance_kwargs=instance_kwargs,
            **kwargs,
        )
        return _reduce_loss(loss, self.reduction)

    def per_sample(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        if obs_cost is None:
            raise ValueError("StandardMSELoss requires obs_cost.")
        loss = self.mse(pred_cost, obs_cost)
        if loss.ndim > 1:
            loss = loss.view(loss.shape[0], -1).mean(dim=1)
        return _normalize_per_sample(loss)


class CosineEmbeddingLoss(nn.Module):
    """Wrapper around nn.CosineEmbeddingLoss with standardized loss signature."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.cosine = nn.CosineEmbeddingLoss(reduction="none")
        self.reduction = reduction

    def forward(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        if obs_cost is None:
            raise ValueError("StandardCosineEmbeddingLoss requires obs_cost.")
        loss = self.per_sample(
            pred_cost,
            obs_cost=obs_cost,
            obs_sol=obs_sol,
            obs_obj=obs_obj,
            instance_kwargs=instance_kwargs,
            **kwargs,
        )
        return _reduce_loss(loss, self.reduction)

    def per_sample(
            self,
            pred_cost: torch.Tensor,
            obs_cost: torch.Tensor | None = None,
            obs_sol: torch.Tensor | None = None,
            obs_obj: torch.Tensor | None = None,
            instance_kwargs: dict | None = None,
            **kwargs,
        ):
        if obs_cost is None:
            raise ValueError("StandardCosineEmbeddingLoss requires obs_cost.")
        target = torch.ones(pred_cost.shape[0], device=pred_cost.device, dtype=pred_cost.dtype)
        loss = self.cosine(pred_cost, obs_cost, target)
        return _normalize_per_sample(loss)
    

# -------------------------------------------------------------------------
# Perturbation Gradient (PG) DCA Loss
# -------------------------------------------------------------------------

class PGDCALoss(nn.Module):
    """
    PG(f_theta(w); c, h, theta0) =
      ( <f_theta(w), x(f_theta0(w))> - <f_theta(w) - h c, x(f_theta(w) - h c)> ) / h

    - pred_cost = f_theta(w) is passed in (computed outside).
    - model is passed in so we can snapshot it periodically as the fixed model0.
    - w is passed in so we can compute f_theta0(w) using the saved model0.
    """
    def __init__(self, optmodel, 
                    h: float,
                    reduction: str="mean", 
                    is_minimization: bool=True,
                    update_every: int = 10, 
                    model0: nn.Module = None):
        super().__init__()
        self.optmodel = optmodel
        self.h = float(h)
        self.reduction = reduction
        self.is_minimization = is_minimization
        self.update_every = int(update_every)

        self.model0 = model0  # frozen snapshot (created lazily)
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _refresh_model0(self, model: nn.Module):
        if self.model0 is None:
            self.model0 = copy.deepcopy(model)
            for p in self.model0.parameters():
                p.requires_grad_(False)
        else:
            self.model0.load_state_dict(model.state_dict())
        self.model0.eval()

    def forward(
        self,
        pred_cost: torch.Tensor,   # f_theta(w) computed outside
        obs_cost: torch.Tensor | None = None,
        obs_sol: torch.Tensor | None = None,
        obs_obj: torch.Tensor | None = None,
        instance_kwargs: dict | None = None,  # per-sample data defining the optimization instance (e.g., feasible region).
        *,
        X: torch.Tensor | None = None,           # inputs to model (needed for model0(X))
        pred_model: nn.Module | None = None,     # live model (for snapshot updates)
        **kwargs,
    ):
        loss = self.per_sample(
            pred_cost,
            obs_cost=obs_cost,
            obs_sol=obs_sol,
            obs_obj=obs_obj,
            instance_kwargs=instance_kwargs,
            X=X,
            pred_model=pred_model,
            **kwargs,
        )
        return _reduce_loss(loss, self.reduction)

    def per_sample(
        self,
        pred_cost: torch.Tensor,   # f_theta(w) computed outside
        obs_cost: torch.Tensor | None = None,
        obs_sol: torch.Tensor | None = None,
        obs_obj: torch.Tensor | None = None,
        instance_kwargs: dict | None = None,  # per-sample data defining the optimization instance (e.g., feasible region).
        *,
        X: torch.Tensor | None = None,           # inputs to model (needed for model0(X))
        pred_model: nn.Module | None = None,     # live model (for snapshot updates)
        **kwargs,
    ):
        t = pred_cost
        y = obs_cost
        h = self.h
        if instance_kwargs is None:
            instance_kwargs = {}
        t_minus = t - h * y

        # ----------------------------------------------------
        # Update / initialize fixed model (theta0)
        # ----------------------------------------------------
        self._step += 1
        if self.model0 is None or (
            self.update_every > 0
            and int(self._step.item()) % self.update_every == 0
        ):
            with torch.no_grad():
                self._refresh_model0(pred_model)

        # ----------------------------------------------------
        # Compute reference term using frozen model and new model
        # ----------------------------------------------------
        with torch.no_grad():
            t0 = self.model0(X)
            x_t0, obj_t0 = self.optmodel(t0, **instance_kwargs)
            x_t0, obj_t0 = x_t0.detach(), obj_t0.detach()
            x_t_minus, obj_t_minus = self.optmodel(t_minus, **instance_kwargs)
            x_t_minus, obj_t_minus = x_t_minus.detach(), obj_t_minus.detach()

        term1 = torch.sum(t * x_t0, axis = 1)

        # Live term: gradients flow through t and through x_fn(t-hc) if x_fn supports autograd
        term2 = torch.sum(t_minus * x_t_minus, axis = 1)

        loss = (term1 - term2) / h
        return _normalize_per_sample(loss)
    

# -------------------------------------------------------------------------
# CILO Loss
# -------------------------------------------------------------------------

class CILOLoss(nn.Module):
    """
    CILO-style loss where the perturbation scale is chosen via CILO_lbda.
    pred_cost is computed externally and passed in.
    """
    def __init__(self, optmodel, 
                    beta: float,
                    reduction: str="mean", 
                    is_minimization: bool=True):
        super().__init__()
        self.optmodel = optmodel
        self.beta = beta
        self.reduction = reduction
        self.is_minimization = is_minimization

    def forward(
        self,
        pred_cost: torch.Tensor,   # f_theta(w) computed outside
        obs_cost: torch.Tensor | None = None,
        obs_sol: torch.Tensor | None = None,
        obs_obj: torch.Tensor | None = None,
        instance_kwargs: dict | None = None,  # per-sample data defining the optimization instance (e.g., feasible region).
        *,
        X: torch.Tensor | None = None,           # inputs to model (needed for model0(X))
        pred_model: nn.Module | None = None,     # live model (for snapshot updates)
        **kwargs,
    ):
        loss = self.per_sample(
            pred_cost,
            obs_cost=obs_cost,
            obs_sol=obs_sol,
            obs_obj=obs_obj,
            instance_kwargs=instance_kwargs,
            X=X,
            pred_model=pred_model,
            **kwargs,
        )
        return _reduce_loss(loss, self.reduction)

    def per_sample(
        self,
        pred_cost: torch.Tensor,   # f_theta(w) computed outside
        obs_cost: torch.Tensor | None = None,
        obs_sol: torch.Tensor | None = None,
        obs_obj: torch.Tensor | None = None,
        instance_kwargs: dict | None = None,  # per-sample data defining the optimization instance (e.g., feasible region).
        *,
        X: torch.Tensor | None = None,           # inputs to model (needed for model0(X))
        pred_model: nn.Module | None = None,     # live model (for snapshot updates)
        **kwargs,
    ):
        t = pred_cost
        y = obs_cost
        if instance_kwargs is None:
            instance_kwargs = {}

        with torch.no_grad():
            h = CILO_lbda(t, y, self.optmodel, self.beta, kwargs=instance_kwargs)

        t_plus = t + h * y

        # ----------------------------------------------------
        # Compute reference term using frozen model and new model
        # ----------------------------------------------------
        with torch.no_grad():
            x_t, obj_t = self.optmodel(t, **instance_kwargs)
            x_t, obj_t = x_t.detach(), obj_t.detach()
            x_t_plus, obj_t_plus = self.optmodel(t_plus, **instance_kwargs)
            x_t_plus, obj_t_plus = x_t_plus.detach(), obj_t_plus.detach()

        term1 = torch.sum(t * x_t_plus, axis = 1)
        # print("term 1: ", term1.mean().item(), torch.sum(x_t_plus, axis = 1).mean().item())

        # Live term: gradients flow through t and through x_fn(t) if x_fn supports autograd
        term2 = torch.sum(t * x_t, axis = 1)
        # print("term 2: ", term2.mean().item(), torch.sum(x_t_plus, axis = 1).mean().item())

        loss = (term1 - term2)
        return _normalize_per_sample(loss)
