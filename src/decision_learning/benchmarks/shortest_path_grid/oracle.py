import numpy as np
import torch


def opt_oracle(costs, size, sens_init=0.0001, sens_upper = 100, epsilon = 1e-6):
    sens = sens_init
    a = sens_init
    b = sens_upper
    sol_size = 2*(size - 1)
    sol, obj = _sp_solver_func(costs, size, sens)
    sol_below_out = torch.sum(1*(torch.sum(sol, axis = 1) < sol_size))
    sol_above_out = torch.sum(1*(torch.sum(sol, axis = 1) > sol_size))
    while(sol_below_out + sol_above_out > 0):
        last_sens = sens
        if sol_below_out > 0:
            sens = (b + sens)*0.5
            a = last_sens
        else:
            sens = (a + sens)*0.5
            b = last_sens
        
        sol, obj = _sp_solver_func(costs, size, sens)
        sol_below_out = torch.sum(1*(torch.sum(sol, axis = 1) < sol_size))
        sol_above_out = torch.sum(1*(torch.sum(sol, axis = 1) > sol_size))

        if abs(sens - last_sens) < epsilon:
            return sol, obj

    return sol, obj


def _sp_solver_func(costs, size, sens = 1e-4):
    """
        Args:
            costs (torch.tensor): a batch of cost vectors 
            size (int): The number n in a n x n node square grid
            sens (float): sensitivty parameter used in computation
        """
    # Forward Pass
    starting_ind = 0
    starting_ind_c = 0
    samples = costs.shape[0]
    V_arr = torch.zeros(samples, size ** 2)
    for i in range(0, 2 * (size - 1)):
        num_nodes = min(i + 1, 2*size - 1 - i)
        num_nodes_next = min(i + 2, 2*size - 1 - i - 1)
        num_arcs = 2 * (max(num_nodes, num_nodes_next) - 1)
        V_1 = V_arr[:, starting_ind:starting_ind + num_nodes]
        layer_costs = costs[:, starting_ind_c:starting_ind_c + num_arcs]
        l_costs = layer_costs[:, 0::2]
        r_costs = layer_costs[:, 1::2]
        next_V_val_l = torch.ones(samples, num_nodes_next) * float('inf')
        next_V_val_r = torch.ones(samples, num_nodes_next) * float('inf')
        if num_nodes_next > num_nodes:
            next_V_val_l[:, :num_nodes_next - 1] = V_1 + l_costs
            next_V_val_r[:, 1:num_nodes_next] = V_1 + r_costs
        else:
            next_V_val_l = V_1[:, :num_nodes_next] + l_costs
            next_V_val_r = V_1[:, 1:num_nodes_next + 1] + r_costs
        next_V_val = torch.minimum(next_V_val_l, next_V_val_r)
        V_arr[:, starting_ind + num_nodes:starting_ind + num_nodes + num_nodes_next] = next_V_val

        starting_ind += num_nodes
        starting_ind_c += num_arcs

    # Backward Pass
    starting_ind = size ** 2
    starting_ind_c = costs.shape[1]
    prev_act = torch.ones(samples, 1)
    sol = torch.zeros(costs.shape)
    for i in range(2 * (size - 1), 0, -1):
        num_nodes = min(i + 1, 2*size - 1 - i)
        num_nodes_next = min(i, 2*size - 1 - i + 1)
        V_1 = V_arr[:, starting_ind - num_nodes:starting_ind]
        V_2 = V_arr[:, starting_ind - num_nodes - num_nodes_next:starting_ind - num_nodes]

        num_arcs = 2 * (max(num_nodes, num_nodes_next) - 1)
        layer_costs = costs[:, starting_ind_c - num_arcs: starting_ind_c]

        if num_nodes < num_nodes_next:
            l_cs_res = ((V_2[:, :num_nodes_next - 1] - V_1 + layer_costs[:, ::2]) < sens) * prev_act
            r_cs_res = ((V_2[:, 1:num_nodes_next] - V_1 + layer_costs[:, 1::2]) < sens) * prev_act
            prev_act = torch.zeros(V_2.shape)
            prev_act[:, :num_nodes_next - 1] += l_cs_res
            prev_act[:, 1:num_nodes_next] += r_cs_res
        else:
            l_cs_res = ((V_2 - V_1[:, :num_nodes - 1] + layer_costs[:, ::2]) < sens) * prev_act[:, :num_nodes - 1]
            r_cs_res = ((V_2 - V_1[:, 1:num_nodes] + layer_costs[:, 1::2]) < sens) * prev_act[:, 1:num_nodes]
            prev_act = torch.zeros(V_2.shape)
            prev_act += l_cs_res
            prev_act += r_cs_res
        cs = torch.zeros(layer_costs.shape)
        cs[:, ::2] = l_cs_res
        cs[:, 1::2] = r_cs_res
        sol[:, starting_ind_c - num_arcs: starting_ind_c] = cs

        starting_ind = starting_ind - num_nodes
        starting_ind_c = starting_ind_c - num_arcs
    # Dimension (samples, num edges)
    obj = torch.sum(sol * costs, axis=1)
    # Dimension (samples, 1)
    return sol.to(torch.float32), obj.reshape(-1,1).to(torch.float32)