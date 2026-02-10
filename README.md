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

base_loss = SPOPlusLoss(optmodel=optmodel, reduction="mean", minimize=True)
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
