import inspect
import time
from functools import partial

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from decision_learning.modeling.val_metrics import decision_regret
from decision_learning.utils import log_runtime, handle_solver

# logging
import logging

logger = logging.getLogger(__name__)
logger.propagate = False
    
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Check if a stream handler already exists
if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # Add the stream handler to the logger
    logger.addHandler(stream_handler)


class GenericDataset(Dataset):
    """Dataset that stores arbitrary tensors keyed by name.

    Use this when losses need custom inputs beyond just `X`. The dataset stores
    the provided tensors and returns a dict per index (plus `instance_kwargs` if
    provided).
    """
    
    def __init__(self, **kwargs):
        """Initialize with named arrays/tensors; converts non-tensors to float32."""
        self.instance_kwargs = kwargs.pop('instance_kwargs', {})
        self.data = {key: torch.as_tensor(value, dtype=torch.float32) if not isinstance(value, torch.Tensor) else value
                     for key, value in kwargs.items()}
        
        self.length = len(next(iter(kwargs.values())))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.data.items()}
        item['instance_kwargs'] = {key: value[idx] for key, value in self.instance_kwargs.items()}
        return item
    
    
def init_loss_data_pretraining(data_dict: dict, 
                            dataloader_params: dict={'batch_size':32, 'shuffle':True}):
    """Build a `GenericDataset` and `DataLoader` from a named data dict.

    `data_dict` must contain `X`. Any additional keys are passed through and
    made available to the loss function.
    """
    if 'X' not in data_dict:
        raise ValueError("data_dict must contain 'X' key")
    
    dataset = GenericDataset(**data_dict)
    dataloader = DataLoader(dataset, **dataloader_params)
    return dataset, dataloader


@log_runtime
def train(pred_model: nn.Module,
    optmodel: callable,
    loss_fn: nn.Module,
    train_data_dict: dict,
    val_data_dict: dict,
    test_data_dict: dict=None,
    dataloader_params: dict={'batch_size':32, 'shuffle':True},
    val_metric: callable=decision_regret,
    device: str='cpu',
    num_epochs: int=10,
    optimizer: torch.optim.Optimizer=None,
    lr: float=1e-2,
    scheduler_params: dict={'step_size': 10, 'gamma': 0.1},
    minimization: bool=True,
    verbose: bool=True,
    ):    
    """Train a prediction model with a decision-focused loss.

    `train_data_dict` and `val_data_dict` must include `X` and any additional
    fields required by `loss_fn`. `val_metric` is partially applied with
    `optmodel`/`minimization` if those appear in its signature.
    """
    # ------------------------- SETUP -------------------------
    if optimizer is None:
        optimizer = torch.optim.Adam(pred_model.parameters(), lr=lr)
        
    scheduler = None
    if scheduler_params is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        
    device = torch.device(device)
    pred_model.to(device)
    logger.info(f"Training on device: {device}")
    
    # bind optmodel/is_minimization if the val metric accepts them
    preset_params = {'optmodel': optmodel, 'is_minimization': minimization}
    preset_params = {k: v for k, v in preset_params.items()
                     if k in inspect.signature(val_metric).parameters}
    val_metric = partial(val_metric, **preset_params)
            
    metrics = []
    
    # DATA SETUP
    train_dataset, train_loader = init_loss_data_pretraining(train_data_dict, dataloader_params)
    val_dataset, val_loader = init_loss_data_pretraining(val_data_dict, dataloader_params)
    
    
    # ------------------------- TRAINING LOOP -------------------------
    for epoch in tqdm(range(num_epochs)):
        epoch_start_time = time.time()
        
        # ------------------------- TRAINING -------------------------
        epoch_losses = []
        pred_model.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, disable=not verbose, desc=f'Training Loader: Epoch {epoch+1}/{num_epochs}')):
            
            # move tensor data to device; leave instance_kwargs on CPU
            for key in batch:
                if not isinstance(batch[key], dict):
                    batch[key] = batch[key].to(device)
            
            pred = pred_model(batch['X'])
            batch['pred_model'] = pred_model
            
            loss = loss_fn(pred, **batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())        
        
        # ------------------------- VALIDATION -------------------------
        pred_model.eval()
        val_regret = calc_test_regret(pred_model=pred_model,
                                test_data_dict=val_data_dict,
                                optmodel=optmodel,
                                is_minimization=minimization                                
                            )                      
        
        test_regret = np.nan
        if test_data_dict is not None:
            test_regret = calc_test_regret(pred_model=pred_model,
                                test_data_dict=test_data_dict,
                                optmodel=optmodel,
                                is_minimization=minimization                                
                            )
        
        # scheduler step
        if scheduler is not None:
            scheduler.step()
            
        epoch_time_sec = time.time() - epoch_start_time
        cur_metric = {'epoch': epoch, 
                    'train_loss': np.mean(epoch_losses), 
                    'val_metric': val_regret,
                    'test_regret': test_regret,
                    'epoch_time_sec': epoch_time_sec}
        metrics.append(cur_metric)
        
        if verbose:
            logger.info(f'epoch: {epoch+1}, train_loss: {np.mean(epoch_losses)}, val_metric: {val_regret}, test_regret: {test_regret}')

        
    metrics = pd.DataFrame(metrics)
    return metrics, pred_model


def calc_test_regret(pred_model: nn.Module, 
                    test_data_dict: dict, 
                    optmodel: callable, 
                    is_minimization: bool=True,                            
                    ):
    """Compute decision regret for a model on a dataset dict.

    Expects `test_data_dict` to contain `X` and either:
    - `cond_exp_cost` and `full_info_obj`, or
    - `obs_cost` and `obs_obj` (fallback benchmark when conditional expectations are unavailable).

    Runs prediction under `torch.no_grad()` and calls `decision_regret`.
    """
    pred_model.eval()
    with torch.no_grad():
        X = torch.tensor(test_data_dict['X'], dtype=torch.float32)
        pred = pred_model(X)
        
        instance_kwargs = test_data_dict.get('instance_kwargs', {})      
        if "cond_exp_cost" in test_data_dict and "full_info_obj" in test_data_dict:
            # print("calc_test_regret: using conditional expectation benchmark (cond_exp_cost/full_info_obj).")
            regret = decision_regret(
                pred,
                cond_exp_cost=test_data_dict["cond_exp_cost"],
                full_info_obj=test_data_dict["full_info_obj"],
                optmodel=optmodel,
                is_minimization=is_minimization,
                instance_kwargs=instance_kwargs,
            )
        else:
            # Without conditional expectations, benchmark against an (unachievable) oracle
            # that can see the costs ahead of time.
            # print("calc_test_regret: using observed-cost fallback benchmark (obs_cost/obs_obj).")
            regret = decision_regret(
                pred,
                cond_exp_cost=test_data_dict["obs_cost"],
                full_info_obj=test_data_dict["obs_obj"],
                optmodel=optmodel,
                is_minimization=is_minimization,
                instance_kwargs=instance_kwargs,
            )
        
    return regret
