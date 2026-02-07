import numpy as np
import torch

from decision_learning.utils import handle_solver


def box_oracle_batch(t: np.ndarray, b: np.ndarray):
    """Solve min <t, z> s.t. 0 <= z <= b in closed form, rowwise."""
    z = b * (t <= 0)
    obj = np.sum(t * z, axis=1)
    return z, obj


def box_oracle_single(t: np.ndarray, b: np.ndarray):
    """Solve min <t, z> s.t. 0 <= z <= b for a single sample."""
    z = b * (t <= 0)
    obj = np.sum(t * z)
    return z, obj


def test_oracle_wrapper_batch_correctness():
    # Exercise the toy box oracle through handle_solver; 
    #verify shape, objective consistency, b sensitivity, and deterministic outputs.
    t = torch.tensor([[-1.0, 2.0], [0.0, -3.0], [1.0, 1.0]])
    b = torch.tensor([[1.0, 5.0], [2.0, 4.0], [3.0, 6.0]])

    sol, obj = handle_solver(
        t,
        optmodel=box_oracle_batch,
        instance_kwargs={"b": b},
        solver_batch_solve=True,
    )

    assert sol.shape == (t.shape[0], t.shape[1])
    expected_obj = np.sum(t.numpy() * sol, axis=1)
    assert np.allclose(obj, expected_obj)
    assert np.array_equal(sol[0], np.array([1.0, 0.0]))
    assert np.array_equal(sol[1], np.array([2.0, 4.0]))
    assert np.isclose(obj[0], -1.0)
    assert np.isclose(obj[1], -12.0)

    sol_repeat, obj_repeat = handle_solver(
        t,
        optmodel=box_oracle_batch,
        instance_kwargs={"b": b},
        solver_batch_solve=True,
    )
    assert np.array_equal(sol, sol_repeat)
    assert np.array_equal(obj, obj_repeat)

    b_changed = b + 1.0
    sol_changed, _ = handle_solver(
        t,
        optmodel=box_oracle_batch,
        instance_kwargs={"b": b_changed},
        solver_batch_solve=True,
    )
    assert not np.array_equal(sol, sol_changed)


def test_oracle_wrapper_single_correctness():
    t = torch.tensor([[-1.0, 2.0], [0.0, -3.0], [1.0, 1.0]])
    b = torch.tensor([[1.0, 5.0], [2.0, 4.0], [3.0, 6.0]])

    sol, obj = handle_solver(
        t,
        optmodel=box_oracle_single,
        instance_kwargs={"b": b},
        solver_batch_solve=False,
    )

    assert sol.shape == (t.shape[0], t.shape[1])
    expected_obj = np.sum(t.numpy() * sol, axis=1)
    assert np.allclose(obj, expected_obj)
    assert np.array_equal(sol[0], np.array([1.0, 0.0]))
    assert np.array_equal(sol[1], np.array([2.0, 4.0]))
    assert np.isclose(obj[0], -1.0)
    assert np.isclose(obj[1], -12.0)

    sol_repeat, obj_repeat = handle_solver(
        t,
        optmodel=box_oracle_single,
        instance_kwargs={"b": b},
        solver_batch_solve=False,
    )
    assert np.array_equal(sol, sol_repeat)
    assert np.array_equal(obj, obj_repeat)

    b_changed = b + 1.0
    sol_changed, _ = handle_solver(
        t,
        optmodel=box_oracle_single,
        instance_kwargs={"b": b_changed},
        solver_batch_solve=False,
    )
    assert not np.array_equal(sol, sol_changed)
