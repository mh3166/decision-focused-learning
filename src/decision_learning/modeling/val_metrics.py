import torch 
import numpy as np
from decision_learning.utils import handle_solver


def decision_regret(pred_cost: torch.tensor, 
        cond_exp_cost: np.ndarray, 
        full_info_obj: np.ndarray,
        optmodel: callable,
        is_minimization: bool=True,        
        instance_kwargs: dict = {}):
    """Compute normalized decision regret for a batch.

    Args:
        pred_cost (torch.tensor): predicted cost parameters.
        cond_exp_cost (np.ndarray | torch.tensor): oracle conditional expectation of cost given features.
        full_info_obj (np.ndarray | torch.tensor): objective values under the conditional-expectation solution.
        optmodel (callable): solver called as optmodel(cost, **instance_kwargs) -> (solution, objective).
        is_minimization (bool): whether the downstream optimization is a minimization problem.
        instance_kwargs (dict): per-sample instance data passed to the solver.

    Notes:
        If the oracle conditional expectation is available, pass it as cond_exp_cost and compute
        full_info_obj via:
            _, full_info_obj = optmodel(cond_exp_cost)
    """
    # get batch's current optimal solution value and objective vvalue based on the predicted cost
    w_hat, z_hat = optmodel(pred_cost, **instance_kwargs)

    # To ensure consistency, convert everything into a pytorch tensor on the same device as pred_cost
    device = pred_cost.device if isinstance(pred_cost, torch.Tensor) else 'cpu'
    w_hat = torch.as_tensor(w_hat, dtype=torch.float32, device=device)
    z_hat = torch.as_tensor(z_hat, dtype=torch.float32, device=device)
    cond_exp_cost = torch.as_tensor(cond_exp_cost, dtype=torch.float32, device=device)
    full_info_obj = torch.as_tensor(full_info_obj, dtype=torch.float32, device=device)
    
    # objective value of pred_cost induced solution (w_hat) based on true cost
    obj_hat = (w_hat * cond_exp_cost).sum(axis=1, keepdim=True)
        
    regret = (obj_hat - full_info_obj).sum()
    if not is_minimization:
        regret = -regret
    
    opt_obj_sum = torch.sum(torch.abs(full_info_obj)).item() + 1e-7
    return regret.item() / opt_obj_sum
        
    
