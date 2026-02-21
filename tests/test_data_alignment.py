import numpy as np

from decision_learning.modeling.pipeline import (
    make_loss_data_dict,
    split_train_val,
)


def _build_alignment_data(num_rows: int) -> dict:
    # Embed a stable per-row ID into every array to detect ordering or alignment drift.
    ids = np.array([f"id-{idx}" for idx in range(num_rows)], dtype=object)
    data = {
        "X": np.stack([ids, ids], axis=1),
        "obs_cost": ids.reshape(-1, 1),
        "obs_sol": np.stack([ids, ids], axis=1),
        "obs_obj": ids.reshape(-1, 1),
        "instance_kwargs": {"size": ids.copy()},
    }
    return data


def _echo_solver(obs_cost, **instance_kwargs):
    # Dummy solver that returns inputs as outputs to preserve row-wise IDs.
    sol = obs_cost.copy()
    obj = obs_cost.copy()
    return sol, obj


def _preprocess_alignment_data(num_rows: int) -> dict:
    # Run the same preprocessing path as the pipeline to validate alignment invariants.
    data = _build_alignment_data(num_rows)
    return make_loss_data_dict(
        X=data["X"],
        obs_cost=data["obs_cost"],
        optmodel=_echo_solver,
        instance_kwargs=data["instance_kwargs"],
    )


def _assert_alignment(split_dict: dict) -> None:
    # Compare row IDs across all outputs; any permutation or mismatch should fail loudly.
    ids_from_x = split_dict["X"][:, 0]
    ids_from_cost = split_dict["obs_cost"][:, 0]
    ids_from_sol = split_dict["obs_sol"][:, 0]
    ids_from_obj = split_dict["obs_obj"][:, 0]
    ids_from_solver = split_dict["instance_kwargs"]["size"]

    assert split_dict["X"].shape[0] == split_dict["obs_cost"].shape[0]
    assert split_dict["X"].shape[0] == split_dict["obs_sol"].shape[0]
    assert split_dict["X"].shape[0] == split_dict["obs_obj"].shape[0]
    assert split_dict["X"].shape[0] == split_dict["instance_kwargs"]["size"].shape[0]

    assert np.array_equal(ids_from_x, ids_from_cost), "X and obs_cost rows misaligned"
    assert np.array_equal(ids_from_x, ids_from_sol), "X and obs_sol rows misaligned"
    assert np.array_equal(ids_from_x, ids_from_obj), "X and obs_obj rows misaligned"
    assert np.array_equal(ids_from_x, ids_from_solver), "X and instance_kwargs rows misaligned"


def test_split_train_valit_preserves_alignment():
    # Baseline check: splitting alone should preserve all row-wise correspondence.
    data = _build_alignment_data(num_rows=12)
    train_dict, val_dict = split_train_val(
        train_d=data,
        val_split_params={"test_size": 0.25, "random_state": 123},
    )

    _assert_alignment(train_dict)
    _assert_alignment(val_dict)


def test_preprocess_and_split_preserve_alignment():
    # Full path check: preprocessing + split should preserve row-wise correspondence.
    preprocessed = _preprocess_alignment_data(num_rows=12)
    train_dict, val_dict = split_train_val(
        train_d=preprocessed,
        val_split_params={"test_size": 0.25, "random_state": 123},
    )

    _assert_alignment(train_dict)
    _assert_alignment(val_dict)
