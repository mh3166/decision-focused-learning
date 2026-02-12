from typing import List, Union
from itertools import product
import copy

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch

from decision_learning.modeling.loss_spec import LossSpec
from decision_learning.utils import handle_solver
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
            pred_model: callable,
            opt_oracle: callable,
            train_instance_kwargs: dict={},
            test_instance_kwargs: dict={},
            train_val_split_params: dict={'test_size':0.2, 'random_state':42},
            loss_specs: List[LossSpec]=[],
            is_minimization: bool=True,
            train_config: dict=None,               
            save_models: bool=False,
            training_loop_verbose: bool=False):
    """Run training across one or more loss specifications.

    Builds train/val/test dicts (including optimal solutions under true costs),
    then trains a fresh copy of `pred_model` for each LossSpec and hyperparameter
    setting in its grid.

    Args:
        X_train, true_cost_train, X_test, true_cost_test: Train/test features and costs.
        pred_model: PyTorch model that predicts costs from features.
        opt_oracle: Callable solver/oracle used to compute optimal solutions/objectives and regret.
        train_instance_kwargs, test_instance_kwargs: Per-sample instance data passed to the oracle.
        train_val_split_params: Dict for `train_test_split` used to create train/val splits.
        loss_specs: List of LossSpec objects defining loss constructors, grids, and extra batch data.
        is_minimization: Whether the downstream optimization is a minimization problem.
        train_config: Training loop configuration (batch size, epochs, lr, scheduler params).
        save_models: Store trained models per loss/hyperparameter if True.
        training_loop_verbose: Verbose training loop logging.

    Usage:
        loss_specs = [
            LossSpec(
                name="PG",
                factory=PGLoss,
                init_kwargs={},
                aux={"optmodel": opt_oracle, "is_minimization": True},
                hyper_grid=expand_hyperparam_grid({...}),
            ),
            LossSpec(name="MSE", factory=MSELoss, init_kwargs={}),
        ]
        metrics, models = run_loss_experiments(..., loss_specs=loss_specs)
    """
    
    # Require at least one loss
    if not loss_specs:
        raise ValueError("Please provide at least one loss specification")
    
    # default training configs. User can override through train_config
    tr_config = {
        'dataloader_params': {'batch_size':32, 'shuffle':True},
        'num_epochs': 10,
        'lr': 0.01,
        'scheduler_params': None
        }
    if train_config is not None:
        tr_config.update(train_config)
        
    
    # -----------------Initial data setup  -----------------
    train_d = make_loss_data_dict(X_train, true_cost_train, opt_oracle, instance_kwargs=train_instance_kwargs)
    train_dict, val_dict = split_train_val(train_d=train_d, val_split_params=train_val_split_params)
    test_data = make_loss_data_dict(X_test, true_cost_test, opt_oracle, instance_kwargs=test_instance_kwargs)
    
    
    #Store Outputs
    overall_metrics = []
    trained_models = {}

    # loop through list of existing and user-defined loss functions
    for loss_idx, spec in enumerate(loss_specs): 
        
        logger.info(f"""Loss number {loss_idx+1}/{len(loss_specs)}, on loss function {spec.name}""")            
        
        train_dict_processed = copy.deepcopy(train_dict)
        val_dict_processed = copy.deepcopy(val_dict)
        test_dict_processed = copy.deepcopy(test_data)

        if spec.extra_batch_data:
            extra_train, extra_val = split_train_val(
                train_d=dict(spec.extra_batch_data),
                val_split_params=train_val_split_params,
            )
            for key in extra_train:
                if key in train_dict_processed or key in val_dict_processed:
                    raise ValueError(f"extra_batch_data key '{key}' collides with existing batch field")
            train_dict_processed.update(extra_train)
            val_dict_processed.update(extra_val)

        grid = spec.hyper_grid or [{}]

        #loop through hyperparameters for loss function
        for idx, hparams in enumerate(grid):
            logger.info(f"""Trial {idx+1}/{len(grid)} for running loss function {spec.name}, current hyperparameters: {hparams}""")            
            
            # INSTANTIATE LOSS FUNCTION
            # Copy the hyperparams to avoid modifying the original dictionary
            orig_param_set = copy.deepcopy(hparams)

            cur_loss = spec.factory(
                **spec.init_kwargs,
                **hparams,
                **spec.aux,
            )

            # -----------------TRAINING LOOP-----------------
            # each loss starts from the same model initialization
            pred_model = copy.deepcopy(pred_model)


            #train
            metrics, trained_model = train(pred_model=pred_model, 
                optmodel=opt_oracle,
                loss_fn=cur_loss,
                train_data_dict=train_dict_processed,
                val_data_dict=val_dict_processed,
                test_data_dict=test_dict_processed,
                minimization=is_minimization,
                verbose=training_loop_verbose,                
                **tr_config)
            metrics['loss_name'] = spec.name            
            metrics['hyperparameters'] = str(orig_param_set)
            overall_metrics.append(metrics)
            
            if save_models: 
                trained_models[spec.name + "_" + str(orig_param_set)] = trained_model
        
    overall_metrics = pd.concat(overall_metrics, ignore_index=True)
    
    return overall_metrics, trained_models
