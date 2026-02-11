# Example of how to locally install and import the code
Navigate to the src folder with `setup.py`
```
conda activate (YOUR_ENVIRONMENT_NAME) # you can choose to skip this step for global install
cd decision-focused-learning/src
pip install -e .
```
now you can import like the jupyter notebook example `decision-focused-learning/notebooks/shortest_path_example.ipynb` or
`decision-focused-learning/notebooks/shortest_path_pipeline_example.ipynb`

```
from decision_learning.data.shortest_path_grid import genData
from decision_learning.modeling.loss import SPOPlusLoss
from decision_learning.modeling.models import LinearRegression
from decision_learning.modeling.train import train, calc_test_regret
```

# Randomized smoothing wrapper
Use `RandomizedSmoothingWrapper` to obtain a perturbation-smoothed version of any loss that implements the standard `per_sample`/`forward` interface.
This replaces the old `SPOPlusLoss(smoothing=...)` behavior.

```python
from decision_learning.modeling.loss import SPOPlusLoss
from decision_learning.modeling.smoothing import RandomizedSmoothingWrapper

base_loss = SPOPlusLoss(optmodel=optmodel, reduction="mean", is_minimization=True)
loss_fn = RandomizedSmoothingWrapper(
    base_loss=base_loss,
    sigma=0.1,
    s=8,
    seed=0,
    antithetic=False,
    control_variate=False,
    reduction="mean",
)

loss = loss_fn(
    pred_cost=pred_cost,
    true_cost=true_cost,
    true_sol=true_sol,
    true_obj=true_obj,
    instance_kwargs=instance_kwargs,
)
```

# Contracts
This repo relies on a small set of interface contracts to keep the training loop simple and extensible.

## Loss function signature
Loss functions must accept extra keyword arguments and ignore anything they don't use.
In practice, this means `forward` (and `per_sample`, if you expose it) should include `**kwargs`.
The training loop passes all batch fields to the loss.

Minimal example:
```python
def forward(self, pred_cost, true_cost=None, **kwargs):
    # ignore unused kwargs
    ...
```

## LossSpec extra batch data
`LossSpec.extra_batch_data` is for per-sample inputs required by a loss. These arrays/tensors must align
with the training examples (first dimension is `n`) and are split into train/val/test with the same
indices as the main dataset, then merged into the batch dict passed to `forward`.

Do not put per-sample tensors into `aux` (that is reserved for fixed objects like solvers, flags, or
reference modules).

Minimal example:
```python
spec = LossSpec(
    name="CustomLoss",
    factory=CustomLoss,
    init_kwargs={},
    extra_batch_data={"extra": extra_tensor},
)
```

## run_loss_experiments with LossSpec
`run_loss_experiments` now accepts a list of `LossSpec` objects directly. You can provide a single
loss or multiple losses, and optionally pass `hyper_grid` for sweeps.

```python
loss_specs = [
    LossSpec(name="PG", factory=PGLoss, init_kwargs={}, aux={"optmodel": optmodel, "is_minimization": True}, hyper_grid=expand_hyperparam_grid({
        "h": [0.1, 0.2],
        "finite_diff_type": ["B", "F"],
    })),
    LossSpec(name="MSE", factory=MSELoss, init_kwargs={}),
]

metrics, models = run_loss_experiments(
    X_train=X_train,
    true_cost_train=true_cost_train,
    X_test=X_test,
    true_cost_test=true_cost_test,
    pred_model=pred_model,
    opt_oracle=optmodel,
    loss_specs=loss_specs,
)
```
