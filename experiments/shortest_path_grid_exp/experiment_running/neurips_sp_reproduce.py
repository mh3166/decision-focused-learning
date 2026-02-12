import importlib
from functools import partial
import time
import os

from torch import nn
import torch
import pandas as pd
import numpy as np

import decision_learning.modeling.pipeline
import decision_learning.benchmarks.shortest_path_grid.data
from decision_learning.utils import handle_solver
from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.pipeline import run_loss_experiments, expand_hyperparam_grid
from decision_learning.modeling.loss_spec import LossSpec
from decision_learning.modeling.loss import (
    SPOPlusLoss,
    MSELoss,
    FYLoss,
    CosineEmbeddingLoss,
    PGLoss,
    CosineSurrogateDotProdVecMagLoss,
    CosineSurrogateDotProdMSELoss,
)
from decision_learning.benchmarks.shortest_path_grid.data import genData
from decision_learning.benchmarks.shortest_path_grid.oracle import opt_oracle

# logging
import logging
logging.basicConfig(level=logging.INFO)


def main():
    
    # ----------------------- SETUP -----------------------
    
    # ----------- Data Setup -----------
    # set random seeds
    torch.manual_seed(105)
    indices_arr = torch.randperm(100000)
    indices_arr_test = torch.randperm(100000)

    # construct experiment data configuration array
    # we'll run 100 trials for each configuration of n and epsilon type
    n_arr = [200, 400, 800, 1600] # number of training data
    ep_arr = ['unif', 'normal'] # epsilon type
    trials = 100

    # generate experiment array
    exp_arr = []
    for n in n_arr:
        for ep in ep_arr:
            for t in range(trials):
                exp_arr.append([n, ep, t])

    # shortest path example data generation configurations
    grid = (5, 5)  # grid size    
    num_feat = 5  # size of feature
    deg = 6  # polynomial degree
    e = .3  # noise width
    
    # path planting for shortest path example - see page 9, subsection "Harder Example with Planted Arcs" in section 4.2 of paper https://arxiv.org/pdf/2402.03256
    planted_good_pwl_params = {'slope0':0, # slope of first segment of piecewise linear cost function for "good" edge cost planted
                        'int0':2, # intercept of first segment of piecewise linear cost function for "good" edge cost planted
                        'slope1':0, # slope of second segment of piecewise linear cost function for "good" edge cost planted
                        'int1':2} # intercept of second segment of piecewise linear cost function for "good" edge cost planted
    planted_bad_pwl_params = {'slope0':4, # slope of first segment of piecewise linear cost function for "bad" edge cost planted
                        'int0':0, # intercept of first segment of piecewise linear cost function for "bad" edge cost planted
                        'slope1':0, # slope of second segment of piecewise linear cost function for "bad" edge cost planted
                        'int1':2.2} # intercept of second segment of piecewise linear cost function for "bad" edge cost planted
    plant_edge = True # to plant edges in shortest path experiment or not
    
    # ----------- optimization model -----------
    optmodel = partial(handle_solver, optmodel=opt_oracle, detach_tensor=False, solver_batch_solve=True)

    # ----------- Results Setup -----------
    save_file = 'shortest_path_neurips_exp_opthandler_check.csv'
    
    # ----------------------- Sequentially loop through experiment trials -----------------------
    # experiment does not take that long (< 10 hours), so we can run it sequentially on HPC instead of parallel processing across multiple compute nodes
         
    for sim in range(len(exp_arr)): 
        start_time = time.time()
        # setup
        exp = exp_arr[sim] # current experiment from experiment array
        num_data = exp[0]  # number of training data
        ep_type = exp[1] # noise type of current experiment
        trial = exp[2] # trial number of current experiment
        logging.info(f"Running experiment: {sim+1}/{len(exp_arr)} with n={num_data}, epsilon type={ep_type}, trial={trial}")
        
        # ------------ Instance specific data ------------
        # training data
        generated_data = genData(num_data=num_data+200, # number of data points to generate for training set
                num_features=num_feat, # number of features 
                grid=grid, # grid shape
                deg=deg, # polynomial degree
                noise_type=ep_type, # epsilon noise type
                noise_width=e, # amount of noise
                seed=indices_arr[trial], # seed the randomness
                plant_edges=plant_edge, # to plant edges or not
                planted_good_pwl_params=planted_good_pwl_params, # cost function for good edges
                planted_bad_pwl_params=planted_bad_pwl_params) # cost function for bad edges

        # testing data
        generated_data_test = genData(num_data=10000, # number of data points to generate for test set
                num_features=num_feat, # number of features 
                grid=grid,  # grid shape
                deg=deg,  # polynomial degree
                noise_type=ep_type,  # epsilon noise type
                noise_width=e, # amount of noise
                seed=indices_arr_test[trial],      # seed the randomness
                plant_edges=plant_edge, # to plant edges or not
                planted_good_pwl_params=planted_good_pwl_params, # cost function for good edges
                planted_bad_pwl_params=planted_bad_pwl_params) # cost function for bad edges
        
        # test solver kwargs
        train_instance_kwargs = {'size': np.zeros(len(generated_data['cost'])) + 5}
        test_instance_kwargs = {'size': np.zeros(len(generated_data_test['cost'])) + 5}
        
        # ------------prediction model------------
        pred_model = LinearRegression(input_dim=generated_data['feat'].shape[1],
                        output_dim=generated_data['cost'].shape[1])
    
        # ------------loss function experiment pipeline------------
        
        # non-PG losses
        preimplement_loss_specs = [
            LossSpec(name='SPO+', factory=SPOPlusLoss, init_kwargs={}, aux={"optmodel": optmodel, "minimize": True}),
            LossSpec(name='MSE', factory=MSELoss, init_kwargs={}),
            LossSpec(name='FY', factory=FYLoss, init_kwargs={}, aux={"optmodel": optmodel, "minimize": True}),
            LossSpec(name='Cosine', factory=CosineEmbeddingLoss, init_kwargs={}),
        ]
        preimplement_loss_results, preimplement_loss_models = run_loss_experiments(X_train=generated_data['feat'],
                true_cost_train=generated_data['cost'],
                X_test=generated_data_test['feat'],
                true_cost_test=generated_data_test['cost_true'], 
                pred_model=pred_model,
                opt_oracle=optmodel,
                train_instance_kwargs=train_instance_kwargs,
                test_instance_kwargs=test_instance_kwargs,
                train_val_split_params={'test_size':200, 'random_state':42},
                loss_specs=preimplement_loss_specs,
                train_config={'num_epochs':100,
                                 'dataloader_params': {'batch_size':32, 'shuffle':True}},
                save_models=True                                                                                 
                )
        
        # PG loss
        pg_loss_specs = [
            LossSpec(
                name='PG',
                factory=PGLoss,
                init_kwargs={},
                aux={"optmodel": optmodel, "minimize": True},
                hyper_grid=expand_hyperparam_grid({
                    'h': [num_data**-.125, num_data**-.25, num_data**-.5, num_data**-1],
                    'finite_diff_type': ['B', 'C', 'F'],
                }),
            )
        ]
        PG_init_results, PG_init_models = run_loss_experiments(X_train=generated_data['feat'],
                true_cost_train=generated_data['cost'],
                X_test=generated_data_test['feat'],
                true_cost_test=generated_data_test['cost_true'], 
                pred_model=preimplement_loss_models['SPO+_{}'],
                opt_oracle=optmodel,
                train_instance_kwargs=train_instance_kwargs,
                test_instance_kwargs=test_instance_kwargs,
                train_val_split_params={'test_size':200, 'random_state':42},
                loss_specs=pg_loss_specs,
                train_config={'num_epochs':100,
                                 'dataloader_params': {'batch_size':32, 'shuffle':True}},
                save_models=False
                )

        # Cosine Surrogate Losses        
        cos_surr_specs = [
            LossSpec(
                name='CosineSurrogateDotProdVecMagLoss',
                factory=CosineSurrogateDotProdVecMagLoss,
                init_kwargs={},
                hyper_grid=expand_hyperparam_grid({'alpha':[0.01, 0.1, 1, 2.5, 5, 7.5, 10]}),
            ),
            LossSpec(
                name='CosineSurrogateDotProdMSELoss',
                factory=CosineSurrogateDotProdMSELoss,
                init_kwargs={},
                hyper_grid=expand_hyperparam_grid({'alpha':[0.01, 0.1, 1, 2.5, 5, 7.5, 10]}),
            ),
        ]
        cos_surr_results, cos_surr_models = run_loss_experiments(X_train=generated_data['feat'],
                        true_cost_train=generated_data['cost'],
                        X_test=generated_data_test['feat'],
                        true_cost_test=generated_data_test['cost_true'], 
                        pred_model=pred_model,
                        opt_oracle=optmodel,
                        train_instance_kwargs=train_instance_kwargs,
                        test_instance_kwargs=test_instance_kwargs,
                        train_val_split_params={'test_size':200, 'random_state':42},
                        loss_specs=cos_surr_specs,
                        train_config={'num_epochs':100,
                                         'dataloader_params': {'batch_size':32, 'shuffle':True}},
                        save_models=False
                        )


        # store current results
        combined_results = pd.concat([preimplement_loss_results, PG_init_results, cos_surr_results], ignore_index=True)
        combined_results['n'] = num_data
        combined_results['eps_type'] = ep_type
        combined_results['trial'] = trial
    
        # write results to file
        if os.path.exists(save_file):
            combined_results.to_csv(save_file, mode='a', header=False, index=False)
            logging.info(f"Appended to {save_file}")
        else:
            combined_results.to_csv(save_file, mode='w', header=True, index=False)
            logging.info(f"Created {save_file}")
            
        # display time taken per trial
        end_time = time.time()
        logging.info(f"Time taken for experiment: {end_time - start_time} seconds")
            

if __name__ == "__main__":
    main()
