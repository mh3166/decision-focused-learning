from typing import List, Union
from itertools import product
import copy

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch

from decision_learning.modeling.loss import get_loss_function
from decision_learning.utils import filter_kwargs, handle_solver
from decision_learning.modeling.train import train

# logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.propagate = False
# Check if a stream handler already exists
if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # Add the stream handler to the logger
    logger.addHandler(stream_handler)
    
    
def make_loss_data_dict(X: Union[np.ndarray, torch.tensor], 
                                true_cost: Union[np.ndarray, torch.tensor], 
                                optmodel: callable,
                                user_def_loss_inputs: dict={},
                                instance_kwargs: dict={}):
    """Build the data dict expected by loss functions.

    Computes the optimal solution/objective under the true cost and returns a
    dictionary with standardized keys: `X`, `true_cost`, `true_sol`,
    `true_obj`, and `instance_kwargs`. Extra inputs for custom losses can be
    merged in via `user_def_loss_inputs`.
    """
    
    sol, obj = optmodel(true_cost, **instance_kwargs)
    final_data = {"X": X, "true_cost": true_cost, "true_sol": sol, "true_obj": obj, "instance_kwargs": instance_kwargs}
    
    if user_def_loss_inputs:
        final_data.update(user_def_loss_inputs)
        
    return final_data


def split_train_val(train_d: dict, val_split_params: dict={'test_size':0.2, 'random_state':42}):
    """Split a training data dict into train/val dicts.

    All non-dict values are split with the same indices. If `instance_kwargs`
    is present, each sub-key is split independently to preserve alignment.
    """
    # ensure deterministic splits across keys
    if 'random_state' not in val_split_params:
        val_split_params['random_state'] = 42
        
    train_dict = {}
    val_dict = {}

    for key, value in train_d.items():
        
        # handle nested instance_kwargs
        if key == 'instance_kwargs':
            train_data = {}
            val_data = {}
            for sol_key, sol_value in value.items():
                train_sol_val, val_sol_value = train_test_split(sol_value, **val_split_params)
                train_data[sol_key] = train_sol_val
                val_data[sol_key] = val_sol_value        
        else:
            train_data, val_data = train_test_split(value, **val_split_params)                                
        
        train_dict[key] = train_data
        val_dict[key] = val_data
        
    return train_dict, val_dict
    
    
def expand_hyperparam_grid(hyperparams: dict[str, list]) -> list[dict]:
    """Return the Cartesian product of hyperparameter lists as dicts."""
    param_names = hyperparams.keys()
    param_values = hyperparams.values()
    
    combinations = list(product(*param_values))
    
    return [dict(zip(param_names, combination)) for combination in combinations] 


def run_loss_experiments(X_train: Union[np.ndarray, torch.tensor], 
            true_cost_train: Union[np.ndarray, torch.tensor],             
            X_test: Union[np.ndarray, torch.tensor],
            true_cost_test: Union[np.ndarray, torch.tensor],            
            predmodel: callable,
            optmodel: callable,
            train_instance_kwargs: dict={},
            test_instance_kwargs: dict={},
            val_split_params: dict={'test_size':0.2, 'random_state':42},
            loss_names: List[str]=[], 
            loss_configs: dict={}, 
            user_defined_loss_inputs: List[dict]=[],
            minimize: bool=True,
            training_configs: dict=None,               
            save_models: bool=False,
            training_loop_verbose: bool=False):
    """Run training across one or more loss functions (and optional grids).

    Builds train/val/test dicts (including optimal solutions under true costs),
    then trains a fresh copy of `predmodel` for each loss and hyperparameter
    setting. Optionally includes user-defined losses.
    """
    
    # Require at least one loss
    if not loss_names and user_defined_loss_inputs is None:
        raise ValueError("Please provide at least one loss function")
    
    # default training configs. User can override through training_configs
    tr_config = {
        'dataloader_params': {'batch_size':32, 'shuffle':True},
        'num_epochs': 10,
        'lr': 0.01,
        'scheduler_params': None
        }
    if training_configs is not None:
        tr_config.update(training_configs)
        
    
    # -----------------Initial data setup  -----------------
    train_d = make_loss_data_dict(X_train, true_cost_train, optmodel, instance_kwargs=train_instance_kwargs)
    train_dict, val_dict = split_train_val(train_d=train_d, val_split_params=val_split_params)
    test_data = make_loss_data_dict(X_test, true_cost_test, optmodel, instance_kwargs=test_instance_kwargs)
    
    
    #Store Outputs
    overall_metrics = []
    trained_models = {}

    # loop through list of existing loss functions
    for loss_idx, loss_n in enumerate(loss_names): 
        
        logger.info(f"""Loss number {loss_idx+1}/{len(loss_names)}, on loss function {loss_n}""")            
        
        cur_loss_fn = get_loss_function(loss_n)
        
        # loss function hyperparameters
        cur_loss_fn_hyperparam_grid = [{}] # equivalent to using default values
        if loss_n in loss_configs:
            cur_loss_fn_hyperparam_grid = expand_hyperparam_grid(loss_configs[loss_n])        
        
        #loop through hyperparameters for loss function
        for idx, param_set in enumerate(cur_loss_fn_hyperparam_grid):
            logger.info(f"""Trial {idx+1}/{len(cur_loss_fn_hyperparam_grid)} for running loss function {loss_n}, current hyperparameters: {param_set}""")            
            
            # INSTANTIATE LOSS FUNCTION
            # Copy the param_set to avoid modifying the original dictionary
            orig_param_set = copy.deepcopy(param_set)
            
            # additional params to add to param_set - optmodel, minimization, etc.
            additional_params = {"optmodel": optmodel, "minimize": minimize}
            param_set.update(additional_params)
            param_set = filter_kwargs(func=cur_loss_fn.__init__, kwargs=param_set)
            cur_loss = cur_loss_fn(**param_set) # instantiate the loss function - optionally with configs if provided            

            # Use a copy to prevent modifications from persisting across loss functions
            train_dict_processed = copy.deepcopy(train_dict)
            
            # -----------------TRAINING LOOP-----------------
            # each loss starts from the same model initialization
            pred_model = copy.deepcopy(predmodel)


            #train
            metrics, trained_model = train(pred_model=pred_model, 
                optmodel=optmodel,
                loss_fn=cur_loss,
                train_data_dict=train_dict_processed,
                val_data_dict=val_dict,
                test_data_dict=test_data,
                minimization=minimize,
                verbose=training_loop_verbose,                
                **tr_config)
            metrics['loss_name'] = loss_n            
            metrics['hyperparameters'] = str(orig_param_set)
            overall_metrics.append(metrics)
            
            if save_models: 
                trained_models[loss_n + "_" + str(orig_param_set)] = trained_model
            
    
    # -----------------USER-DEFINED LOSS FUNCTIONS-----------------
    for idx, user_defined_loss_input in enumerate(user_defined_loss_inputs):
        
        logger.info(f"""Trial {idx+1}/{len(user_defined_loss_inputs)} for user-defined loss functions, current loss function: {user_defined_loss_input['loss_name']}""")   
        
        cur_loss = user_defined_loss_input['loss']()
        
        # TODO: allow grids for user-defined losses
        pred_model = copy.deepcopy(predmodel)
        
        # -----------------Initial data setup for user-defined loss functions-----------------
        user_def_loss_train_d = make_loss_data_dict(X=X_train,
                                true_cost=true_cost_train, 
                                optmodel=optmodel,
                                user_def_loss_inputs=user_defined_loss_input['data'], # user-provided data
                                instance_kwargs=train_instance_kwargs)    
        train_dict, val_dict = split_train_val(train_d=user_def_loss_train_d, val_split_params=val_split_params)        
        
        #train
        metrics, trained_model = train(pred_model=pred_model,
            optmodel=optmodel,
            loss_fn=cur_loss,
            train_data_dict=train_dict,
            val_data_dict=val_dict,
            test_data_dict=test_data,
            minimization=minimize,
            verbose=training_loop_verbose,
            **tr_config)

        metrics['loss_name'] = user_defined_loss_input['loss_name']
        metrics['hyperparameters'] = None            
        overall_metrics.append(metrics)
        
        if save_models: # store trained_model under loss_name and hyperparameters
            trained_models[user_defined_loss_input['loss_name']] = trained_model
        
    overall_metrics = pd.concat(overall_metrics, ignore_index=True)
    
    return overall_metrics, trained_models
