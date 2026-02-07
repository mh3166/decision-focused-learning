import inspect
import time
from functools import wraps

import numpy as np
import torch 
import numpy as np

import logging 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handle_solver(pred_cost: torch.tensor,                   
                optmodel: callable, 
                instance_kwargs: dict={},                
                detach_tensor: bool=True,
                solver_batch_solve: bool=False):
    """Wrapper function to handle calling the optimization model solver. It will handle the following:
    1. Detach the tensors and convert them to numpy arrays if needed
    2. If solver_batch_solve is True, it will passs the entire batch of data to the optimization model solver 
    otherwise it will call the optimization model solver for each data point in the batch
    
    Handling instance_kwargs:
    We expect instance_kwargs to be a dictionary of per-sample arrays of data that define each instance and that the solver may need, 
    ex: {'coef_matrix': [B, N, M]}, where B is the batch size, N is the number of rows in the matrix, and M is the number of columns in the matrix.
        In this case, instance_kwargs['coef_matrix'][i] will be the coef_matrix for the i-th data point in the batch.                    
    In this case, we will filter out the invalid arguments for the optimization model solver, 
    and then depending on solver_batch_solve:
        - if solver_batch_solve is True, we will pass the entire batch of data to the optimization model solver
        - if solver_batch_solve is False, we will call the optimization model solver for each data point in the batch
            by extracting the i-th data point from instance_kwargs for each key in instance_kwargs
            
    Args:
        optmodel (callable): optimization model
        pred_cost (dict): predicted coefficients/parameters for optimization model
        instance_kwargs (dict): a dictionary of per-sample arrays of data that define each instance and that the solver
            may need to solve the optimization model. For example, instance_kwargs could look like:
            {'coef_matrix': [B, N, M]}, where B is the batch size, N is the number of rows in the matrix, and M is the number of columns in the matrix.
            In this case, instance_kwargs['coef_matrix'][i] will be the coef_matrix for the i-th data point in the batch.                    
        detach_tensor (bool): whether to detach the tensors and convert them to numpy arrays
        solver_batch_solve (bool): whether to pass the entire batch of data to the optimization model solver

    Returns:
        tuple: optimal solution value and objective value based on the predicted cost
    """
    if detach_tensor:
        pred_cost = pred_cost.detach().cpu().numpy()
        
        for key in instance_kwargs:
            if isinstance(instance_kwargs[key], torch.Tensor):
                instance_kwargs[key] = instance_kwargs[key].detach().cpu().numpy()

    # double check to ensure instance_kwargs only contains valid arguments for optmodel          
    instance_kwargs = filter_kwargs(optmodel, instance_kwargs)    
    if solver_batch_solve:                
        sol, obj = optmodel(pred_cost, **instance_kwargs)
        
    else:
        # if solver is not batch solve, we will call the solver for each data point in the batch
        sol = []
        obj = []
        for i in range(pred_cost.shape[0]): 
            # extract the i-th data point from instance_kwargs for each key in instance_kwargs
            cur_instance_kwargs = {k: v[i] for k, v in instance_kwargs.items()}
            
            sol_i, obj_i = optmodel(pred_cost[i], **cur_instance_kwargs)
            sol.append(sol_i)
            obj.append(obj_i)
        sol = np.array(sol)
        obj = np.array(obj)
    
    
    return sol, obj
    

def filter_kwargs(func: callable, kwargs: dict) -> dict:
    """Filter out the valid arguments for a function from a dictionary of arguments. This is useful when you want to
    pass a dictionary of arguments to a function, but only want to pass the valid arguments to the function. 

    Args:
        func (callable): function to filter arguments for
        kwargs (dict): dictionary of arguments to filter

    Returns:
        dict: dictionary of valid arguments for the function
    """
    signature = inspect.signature(func) # get the signature of the function
    valid_args = {key: value for key, value in kwargs.items() if key in signature.parameters} # filter out invalid args
    return valid_args


def log_runtime(func):
    """Decorator to log the runtime of a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function '{func.__name__}' took {end_time - start_time} seconds to run.")
        return result
    return wrapper

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
