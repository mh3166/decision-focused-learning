from pyepo.model.opt import optModel
from pyepo.func.abcmodule import optModule
from pyepo import EPO
from gurobipy import GRB
import gurobipy as gp
from copy import copy
import copy as cpy2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pyepo
from torch.utils.data import DataLoader
from torch.autograd import Function
import time
import pandas as pd
from torch import nn
import sys

from pyepo.model.grb.grbmodel import optGrbModel

# def genData_portfolio(num_data, dat_Y, dat_X, seed=135):
# #add some new noise to Y
#     np.random.RandomState(seed)
#     n = num_data
#     d = dat_Y.shape[1]
#     c = np.zeros((n, d))
#     x = np.zeros((n, dat_X.shape[1]))
#     c_hat = np.zeros((n, d))
#     for i in range(n):
#         rand_indx = np.random.choice(range(120))
#         c[i, :] = dat_Y[rand_indx, :]
#         x[i, :] = dat_X[rand_indx, :]
#         c_hat[i, :] = c[i, :] + np.random.multivariate_normal(mean=np.zeros(12), cov=Sigma * noise)
#         x[i, 1:] += np.random.multivariate_normal(mean=np.zeros(12), cov=Sigma * noise)
#     return torch.FloatTensor(x), torch.FloatTensor(c), torch.FloatTensor(c_hat)


#returns (X, Y) as numpy arrays
def gen_data_sub(num_samples, support_Y, support_X, Sigma, seed):
    num_support_pts = support_Y.shape[0]
    d = support_Y.shape[1]
    np.random.seed(seed) 
    rnd_scenarios = np.random.randint(num_support_pts, size=num_samples)
    Y = support_Y[rnd_scenarios, :]
    X = support_X[rnd_scenarios].copy()  # copy seems necessary to avoid overwriting support
    X[:, 1:] += np.random.multivariate_normal(np.zeros(d), Sigma, size=num_samples)

    return X, Y

#sqrt_det_Sigma need only be correct up to a scaling
#returns (X, fstar(X), Y) as Tensors 
def genData_portfolio_2(num_data, dat_Y, dat_X, Sigma, Sigma_inv, sqrt_det_Sigma, seed):
    X, Y = gen_data_sub(num_data, dat_Y, dat_X, Sigma, seed)
    f_out = np.zeros_like(Y)
    for i in range(num_data):
        f_out[i, :] = fstar(X[i, :], dat_Y, dat_X, Sigma_inv, sqrt_det_Sigma)
    return torch.FloatTensor(X), torch.FloatTensor(f_out), torch.FloatTensor(Y)


# returns the multivariate normal density UP TO PROPORTIONALITY constant
def mvn_density(x, mu, Sigma_inv, sqrt_det_Sigma):
    return (np.exp(-0.5 * (x - mu) @ Sigma_inv @ (x - mu)) / sqrt_det_Sigma)


# #computes E[Y | X ] under above model.
# strictly speaking sqrt_det_Sigma only needs to be specified up to a proportionality constant
def fstar(X, support_Y, support_X, Sigma_inv, sqrt_det_Sigma):
    # compute the normalizing coefficient
    # for now do this in a loop
    prob = 0.
    cond_exp = 0.
    for i in range(support_Y.shape[0]):
        density = mvn_density(X[1:], support_X[i, 1:], Sigma_inv, sqrt_det_Sigma)
        prob += density
        cond_exp += density * support_Y[i, :]

    return cond_exp / prob

def regret_func(predmodel, x, c, optmodel, or_perf_mean):
    """
    A function to evaluate model performance with normalized true regret

    Args:
        predmodel (nn): a regression neural network for cost prediction
        optmodel (optModel): an PyEPO optimization model
        dataloader (DataLoader): Torch dataloader from optDataSet

    Returns:
        float: true regret loss
    """
    # evaluate
    predmodel.eval()
    with torch.no_grad():  # no grad
        cp = predmodel(x).to("cpu").detach().numpy()

    sol_cp, obj_cp = _solve_in_pass(cp, optmodel)
    out_obj_cp = torch.mean(sol_cp * c)

    rel_ratio = (out_obj_cp.item() - or_perf_mean)/(abs(or_perf_mean) + 1e-10)

    #return (out_obj_cp.item() - or_perf_mean)/(abs(or_perf_mean) + 1e-10), torch.mean(sol_cp * c).item()
    return rel_ratio, out_obj_cp.item()

#returns both the sum of oracle performance in batch and mean performance?
#@VG why?
def compute_oracle(c, optmodel):
    sol_c, obj_c = _solve_in_pass(c, optmodel)
    return torch.sum(sol_c * c).item(), torch.mean(sol_c * c).item()

# prediction model
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(13, 12)

    def forward(self, x):
        out = self.linear(x)
        return out

def _solve_in_pass(cp, optmodel):
    ins_num = len(cp)
    sol = []
    obj = []
    for i in range(ins_num):
        # solve
        optmodel.setObj(cp[i])
        solp, objp = optmodel.solve()
        sol.append(solp)
        obj.append(objp)
    # to torch
    sol = torch.tensor(sol)
    obj = torch.tensor(obj)
    return sol, obj

class portfolioModel(optGrbModel):
    """
    This class is optimization model for portfolio optimization problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        num_assets (int): number of assets in portfolio
        gamma (float): the threshold for variance
        returns
    """

    def __init__(self, Sigma, gamma):
        """
        Args:
            Sigma (np.ndarray): covariance matrix (known) of returns
            gamma (float): risk threshold for variance
        """
        self.num_assets = Sigma.shape[0]
        self.Sigma = np.array(Sigma)
        self.gamma = gamma
        super().__init__()

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("portfolio")
        # variables
        x = m.addMVar(self.num_assets, name="x", vtype=GRB.CONTINUOUS, lb = 0, ub = 1)
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints
        m.addConstr(x @ self.Sigma @ x <= self.gamma)
        m.addConstr(x.sum() <= 1)
        return m, x

class SPOPlus2(optModule):
    """
    An autograd module for SPO+ Loss, as a surrogate loss function of SPO Loss,
    which measures the decision error of the optimization problem.

    For SPO/SPO+ Loss, the objective function is linear and constraints are
    known and fixed, but the cost vector needs to be predicted from contextual
    data.

    The SPO+ Loss is convex with subgradient. Thus, it allows us to design an
    algorithm based on stochastic gradient descent.

    Reference: <https://doi.org/10.1287/mnsc.2020.3922>
    """

    def __init__(self, optmodel, processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # build carterion
        self.spop = SPOPlusFunc()

    def forward(self, pred_cost, true_cost, true_sol, true_obj):
        """
        Forward pass
        """
        loss = self.spop.apply(pred_cost, true_cost, true_sol, true_obj, self)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss

class SPOPlusFunc(Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(ctx, pred_cost, true_cost, true_sol, true_obj, module):
        """
        Forward pass for SPO+

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values
            module (optModule): SPOPlus modeul

        Returns:
            torch.tensor: SPO+ loss
        """
        # get device
        device = pred_cost.device
        # convert tensor
        cp = pred_cost.detach().to("cpu")
        c = true_cost.detach().to("cpu")
        w = true_sol.detach().to("cpu")
        z = true_obj.detach().to("cpu")
        # check sol
        #_check_sol(c, w, z)
        # solve
        # sol, obj = _solve_or_cache(2 * cp - c, module)
        # module.optmodel.setObj(2 * cp - c)
        # sol, obj = module.optmodel.solve()
        sol, obj = _solve_in_pass(2 * cp - c, module.optmodel)
        # calculate loss
        loss = - obj.reshape(-1,1) + 2 * torch.sum(cp * w, axis = 1).reshape(-1,1) - z.reshape(-1,1)
        # sense
        if module.optmodel.modelSense == EPO.MAXIMIZE:
            loss = - loss
        # convert to tensor
        # save solutions
        ctx.save_for_backward(true_sol, sol)
        # add other objects to ctx
        ctx.optmodel = module.optmodel
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        w, wq = ctx.saved_tensors
        optmodel = ctx.optmodel
        if optmodel.modelSense == EPO.MINIMIZE:
            grad = 2 * (w - wq)
        if optmodel.modelSense == EPO.MAXIMIZE:
            grad = 2 * (wq - w)
        return grad_output * grad, None, None, None, None

class PG_Loss(optModule):
    """
    An autograd module for PG Loss, as a surrogate loss function of SPO Loss,
    which measures the decision error of the optimization problem.

    For SPO/PG Loss, the objective function is linear and constraints are
    known and fixed, but the cost vector needs to be predicted from contextual
    data.

    The PG Loss is non-convex but with Clarke Differentials. Thus, it allows us to design an
    algorithm based on SGD.
    """

    def __init__(self, optmodel, h=1, finite_diff_type='B', processes=1, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # build carterion
        self.spop = PGLossFunc()
        self.h = h
        self.finite_diff_type = finite_diff_type

    def forward(self, pred_cost, true_cost):
        """
        Forward pass
        """
        loss = self.spop.apply(pred_cost, true_cost, self.h, self.finite_diff_type, self)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = torch.mean(loss)
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss


class PGLossFunc(Function):
    """
    A autograd function for SPO+ Loss
    """

    @staticmethod
    def forward(ctx, pred_cost, true_cost, h, finite_diff_type, module):
        """
        Forward pass for SPO+

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_cost (torch.tensor): a batch of true values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            true_obj (torch.tensor): a batch of true optimal objective values
            module (optModule): SPOPlus modeul

        Returns:
            torch.tensor: SPO+ loss
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu")
        c = true_cost.detach().to("cpu")

        if finite_diff_type == 'C':
            cp_plus = cp + h * c
            cp_minus = cp - h * c
            step_size = 1 / (2 * h)
        elif finite_diff_type == 'B':
            cp_plus = cp
            cp_minus = cp - h * c
            step_size = 1 / h
        elif finite_diff_type == 'F':
            cp_plus = cp + h * c
            cp_minus = cp
            step_size = 1 / h

        sol_plus, obj_plus = _solve_in_pass(cp_plus, module.optmodel)
        sol_minus, obj_minus = _solve_in_pass(cp_minus, module.optmodel)

        # calculate loss
        loss = (obj_plus.reshape(-1,1) - obj_minus.reshape(-1,1)) * step_size

        # sense
        if module.optmodel.modelSense == EPO.MAXIMIZE:
            loss = - loss

        # convert to tensor
        # save solutions
        ctx.save_for_backward(sol_plus, sol_minus)
        # add other objects to ctx
        ctx.optmodel = module.optmodel
        ctx.step_size = step_size
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for SPO+
        """
        sol_plus, sol_minus = ctx.saved_tensors
        optmodel = ctx.optmodel
        step_size = ctx.step_size

        grad = step_size * (sol_plus - sol_minus)

        return grad_output * grad, None, None, None, None

class perturbedFenchelYoung(optModule):
    """
    An autograd module for Fenchel-Young loss using perturbation techniques. The
    use of the loss improves the algorithmic by the specific expression of the
    gradients of the loss.

    For the perturbed optimizer, the cost vector need to be predicted from
    contextual data and are perturbed with Gaussian noise.

    The Fenchel-Young loss allows to directly optimize a loss between the features
    and solutions with less computation. Thus, allows us to design an algorithm
    based on stochastic gradient descent.

    Reference: <https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, processes=1,
                 seed=135, solve_ratio=1, reduction="mean", dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            seed (int): random state seed
            solve_ratio (float): the ratio of new solutions computed during training
            reduction (str): the reduction to apply to the output
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, reduction, dataset)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # random state
        self.rnd = np.random.RandomState(seed)
        # build optimizer
        self.pfy = perturbedFenchelYoungFunc()

    def forward(self, pred_cost, true_sol):
        """
        Forward pass
        """
        loss = self.pfy.apply(pred_cost, true_sol, self)
        # reduction
        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(self.reduction))
        return loss

class perturbedFenchelYoungFunc(Function):
    """
    A autograd function for Fenchel-Young loss using perturbation techniques.
    """

    @staticmethod
    def forward(ctx, pred_cost, true_sol, module):
        """
        Forward pass for perturbed Fenchel-Young loss

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            module (optModule): perturbedFenchelYoung module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        w = true_sol.detach().to("cpu")
        # sample perturbations
        noises = module.rnd.normal(0, 1, size=(module.n_samples, *cp.shape))

        ptb_c = cp + module.sigma * noises
        ptb_c = ptb_c.reshape(-1, noises.shape[2])
        # solve with perturbation
        # ptb_sols, ptb_obj = _solve_or_cache(ptb_c, module)
        # module.optmodel.setObj(ptb_c)
        # ptb_sols, ptb_obj = module.optmodel.solve()
        ptb_sols, ptb_obj = _solve_in_pass(ptb_c, optmodel)

        ptb_sols = ptb_sols.reshape(module.n_samples, -1, ptb_sols.shape[1])
        # solution expectation
        e_sol = ptb_sols.mean(axis=0)

        # ptb_c = cp + module.sigma * noises
        # solve with perturbation
        # ptb_sols = _solve_or_cache(ptb_c, module)
        # solution expectation
        # e_sol = ptb_sols.mean(axis=1)
        # difference
        if module.optmodel.modelSense == EPO.MINIMIZE:
            diff = w - e_sol
        if module.optmodel.modelSense == EPO.MAXIMIZE:
            diff = e_sol - w
        # loss
        loss = torch.sum(diff**2, axis=1)
        # convert to tensor
        # diff = torch.FloatTensor(diff).to(device)
        # loss = torch.FloatTensor(loss).to(device)
        # save solutions
        ctx.save_for_backward(diff)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for perturbed Fenchel-Young loss
        """
        grad, = ctx.saved_tensors
        grad_output = torch.unsqueeze(grad_output, dim=-1)
        return grad * grad_output, None, None


def trainModel(reg, loss_func, loss_name, optmodel, or_perf_mean, val_or_perf_mean, loader_train, val_x, val_c, test_x, test_c, trial, num_data, use_gpu=False, num_epochs=100, lr=1e-2,
               h_schedule=False, lr_schedule=False, early_stopping_cfg=None):
    # set adam optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)

    # train mode
    reg.train()
    # init log
    loss_log = [['trial', 'n', 'epoch', 'rand_start', 'h', 'loss_name', 'regret', 'val_regret', 'return', 'val_return', 'or_return', 'val_or_return']]
    if 'PG' in loss_name:
        num_rand_starts = 1
        h = loss_func.h
    elif loss_name == "DCA":
        num_rand_starts = 1
        h = loss_func.h
    else:
        num_rand_starts = 1
        h = 0
    # init elpased time
    with torch.no_grad():
        predmodel_0 = LinearRegression()
        predmodel_0.linear.weight.copy_(torch.nn.Parameter(reg.linear.weight.detach(), requires_grad=False))
    for r in range(num_rand_starts):
        # reg.apply(weights_init)
        for epoch in range(num_epochs):
            # load data
            batch_loss = [0]
            if epoch % 30 == 0:
                with torch.no_grad():
                    predmodel_0 = LinearRegression()
                    predmodel_0.linear.weight.copy_(torch.nn.Parameter(reg.linear.weight.detach(), requires_grad=False))
                    # predmodel_0.linear.weight = torch.nn.Parameter(reg.linear.weight.detach(), requires_grad=False)
                    predmodel_0.eval()
            for i, data in enumerate(loader_train):
                x, c, w, z = data
                w = w.reshape(w.shape[0], -1)
                z = z.reshape(z.shape[0], -1)
                with torch.no_grad():
                    predmodel_0.eval()
                    cp_0 = predmodel_0(x)
                # cuda
                if use_gpu == True:
                    x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
                # forward pass
                cp = reg(x)

                if loss_name in ['SPO+']:
                    loss = loss_func(cp, c, w, z)
                elif loss_name in ['PGB', 'PGF', 'PGC', 'MSE', 'DBB', 'LTR_pair', 'LTR_point', 'LTR_list']:
                    loss = loss_func(cp, c)
                elif loss_name in ['DCA']:
                    loss = loss_func(cp, cp_0, c)
                elif loss_name in ['FYL']:
                    loss = loss_func(cp, w)

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            train_regret = sum(batch_loss)/num_data
            regret, return_perf = regret_func(reg, torch.FloatTensor(test_x), torch.FloatTensor(test_c), optmodel, or_perf_mean)
            val_regret, val_return_perf = regret_func(reg, torch.FloatTensor(val_x), torch.FloatTensor(val_c), optmodel, val_or_perf_mean)

            loss_log.append([trial, num_data, epoch, r, h, loss_name,
                             regret, val_regret,
                             return_perf, val_return_perf,
                             or_perf_mean, val_or_perf_mean])

            print(
                "Epoch {:2},  Train_Regret: {:7.4f}%, Val_Regret: {:7.4f}%, Regret: {:7.4f}%".format(epoch + 1, train_regret * 100,
                                                                                                   val_regret * 100,
                                                                                                   regret * 100))

    return loss_log, predmodel_0


####################
# actually run the experiment

torch.manual_seed(105)
indices_arr = torch.randperm(100000)
indices_arr_test = torch.randperm(100000)

sim = int(sys.argv[1])

n_arr = [200, 400, 800, 1600]
trials = 50

exp_arr = []
for n in n_arr:
    for t in range(trials):
        exp_arr.append([n, t])

exp = exp_arr[sim]

#load up the covariance matrix from the file
Sigma = np.loadtxt('./cov_120.csv', delimiter=',', skiprows=1)
sqrt_det_Sigma = np.sqrt(np.linalg.det(Sigma))
Sigma_inv = np.linalg.inv(Sigma)

#load up the support for rvs 
dat_120 = np.loadtxt('./dat120_withlags.csv', delimiter=',', skiprows=1)
noise = 0.1
num_data = exp[0]
trial = exp[1]
epochs = 100
vol_scaling = 0.5

#split into X, Y for ease
dat_Y = -dat_120[:, 1:13]  #introduce the minus sign so that we are minimizing NEGATIVE returns.  
dat_X = dat_120[:, 13:]

#prepend a 1 for convenience
dat_X = np.hstack((np.ones((dat_X.shape[0], 1)), dat_X))

#we use vol_scaling here to nonise up the data further
#note sqrt_det_Sigma should also scale, but way we do computation is immune to multiplicative error
feat, cost_true, cost = genData_portfolio_2(num_data + 200, dat_Y, dat_X, Sigma * vol_scaling, Sigma_inv/vol_scaling, sqrt_det_Sigma, seed=indices_arr[sim].item())

x_train, x_val, c_train, c_val = train_test_split(feat, cost, test_size=200, random_state=indices_arr_test[sim].item())
x_test, c_test, c_hat_test = genData_portfolio_2(2000, dat_Y, dat_X, Sigma * vol_scaling, Sigma_inv/vol_scaling, sqrt_det_Sigma, seed = 1000)

optmodel = portfolioModel(Sigma, 0.1)
dataset = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

or_perf_sum, or_perf_mean = compute_oracle(c_test, optmodel)
val_or_perf_sum, val_or_perf_mean = compute_oracle(c_val, optmodel)

spop_loss_func = SPOPlus2(optmodel)
# init prediction model
predmodel = LinearRegression()
spop_out, spop_reg = trainModel(predmodel, spop_loss_func, 'SPO+', optmodel, or_perf_mean, val_or_perf_mean, dataloader, x_val, c_val, x_test, c_test, trial, num_data,
                    use_gpu=False, num_epochs=epochs, lr=1e-2,
                    h_schedule=False, lr_schedule=False, early_stopping_cfg=None)
spop_reg_pgb = cpy2.deepcopy(spop_reg)
# spop_reg_pgf = cpy2.deepcopy(spop_reg)
spop_reg_pgc = cpy2.deepcopy(spop_reg)
# spop_reg_dca = cpy2.deepcopy(spop_reg)
spop_df = pd.DataFrame(columns=spop_out[0], data=spop_out[1:])
#
print("FYL")
fy_loss_func = perturbedFenchelYoung(optmodel)
# init prediction model
predmodel = LinearRegression()
fy_out, fy_reg = trainModel(predmodel, fy_loss_func, 'FYL', optmodel, or_perf_mean, val_or_perf_mean, dataloader, x_val, c_val, x_test, c_test, trial, num_data,
                    use_gpu=False, num_epochs=epochs, lr=1e-2,
                    h_schedule=False, lr_schedule=False, early_stopping_cfg=None)
fy_df = pd.DataFrame(columns=fy_out[0], data=fy_out[1:])

print("MSE")
mse_loss_func = nn.MSELoss()
# init prediction model
predmodel = LinearRegression()
mse_out, mse_reg = trainModel(predmodel, mse_loss_func, 'MSE', optmodel, or_perf_mean, val_or_perf_mean, dataloader, x_val, c_val, x_test, c_test, trial, num_data,
                    use_gpu=False, num_epochs=epochs, lr=1e-2,
                    h_schedule=False, lr_schedule=False, early_stopping_cfg=None)
mse_df = pd.DataFrame(columns=mse_out[0], data=mse_out[1:])

df_arr = [spop_df, fy_df, mse_df]

h_arr = [num_data**-.125, num_data**-.25, num_data**-.5, num_data**-1]
for h in h_arr:
    pgb_loss_func = PG_Loss(optmodel, h=h, finite_diff_type='B')
    # pgb_loss_func = PG_Loss(optmodel, h=h, finite_diff_type='B')
    # init prediction model
    predmodel_b = LinearRegression()
    predmodel_b.linear.weight = spop_reg_pgb.linear.weight

    pgb_out, pgb_reg = trainModel(predmodel_b, pgb_loss_func, 'PGB', optmodel, or_perf_mean, val_or_perf_mean, dataloader, x_val, c_val, x_test, c_test,
                                  trial, num_data,
                                  use_gpu=False, num_epochs=epochs, lr=1e-2,
                                  h_schedule=False, lr_schedule=False, early_stopping_cfg=None)
    pgb_df = pd.DataFrame(columns=pgb_out[0], data=pgb_out[1:])
    df_arr.append(pgb_df)

    # PGC
    pgc_loss_func = PG_Loss(optmodel, h=h, finite_diff_type='C')
    # init prediction model
    predmodel_c = LinearRegression()
    predmodel_c.linear.weight = spop_reg_pgc.linear.weight

    pgc_out, pgc_reg = trainModel(predmodel_c, pgc_loss_func, 'PGC', optmodel, or_perf_mean, val_or_perf_mean, dataloader, x_val, c_val,
                                  x_test, c_test, trial, num_data,
                                  use_gpu=False, num_epochs=epochs, lr=1e-2,
                                  h_schedule=False, lr_schedule=False, early_stopping_cfg=None)
    pgc_df = pd.DataFrame(columns=pgc_out[0], data=pgc_out[1:])
    df_arr.append(pgc_df)

df_all = pd.concat(df_arr)
df_all.to_csv("port_experiment_" + str(sim) + ".csv", index=False)