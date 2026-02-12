import torch

from decision_learning.modeling.smoothing import RandomizedSmoothingFunc


def test_repeat_interleave_nested_tuple_with_batched_instance_kwargs():
    B = 3
    d = 2
    s = 4

    true_cost = torch.arange(B * d).float().view(B, d)
    true_sol = torch.arange(100, 100 + B).float().view(B, 1)
    instance_kwargs = {
        "b": torch.arange(200, 200 + B).float(),
        "c": torch.arange(300, 300 + B * 3).float().view(B, 3),
    }
    h = 0.7

    loss_args = (true_cost, true_sol, instance_kwargs, h)

    expanded = RandomizedSmoothingFunc.repeat_interleave_nested(loss_args, s, dim=0)

    assert expanded[0].shape == (B * s, d)
    assert torch.equal(expanded[0], true_cost.repeat_interleave(s, dim=0))

    assert expanded[1].shape == (B * s, 1)
    assert torch.equal(expanded[1], true_sol.repeat_interleave(s, dim=0))

    assert isinstance(expanded[2], dict)
    assert expanded[2]["b"].shape == (B * s,)
    assert torch.equal(expanded[2]["b"], instance_kwargs["b"].repeat_interleave(s, dim=0))

    assert expanded[2]["c"].shape == (B * s, 3)
    assert torch.equal(expanded[2]["c"], instance_kwargs["c"].repeat_interleave(s, dim=0))

    assert expanded[3] == h


def test_repeat_interleave_nested_dict_with_instance_kwargs():
    B = 3
    d = 2
    s = 4

    true_cost = torch.arange(B * d).float().view(B, d)
    true_sol = torch.arange(100, 100 + B).float().view(B, 1)
    true_obj = torch.arange(500, 500 + B).float()
    instance_kwargs = {
        "b": torch.arange(200, 200 + B).float(),
        "c": torch.arange(300, 300 + B * 3).float().view(B, 3),
    }
    h = 0.7

    loss_kwargs = {
        "true_cost": true_cost,
        "true_sol": true_sol,
        "true_obj": true_obj,
        "instance_kwargs": instance_kwargs,
        "h": h,
    }

    expanded = RandomizedSmoothingFunc.repeat_interleave_nested(loss_kwargs, s, dim=0)

    assert expanded["true_cost"].shape == (B * s, d)
    assert torch.equal(expanded["true_cost"], true_cost.repeat_interleave(s, dim=0))

    assert expanded["true_sol"].shape == (B * s, 1)
    assert torch.equal(expanded["true_sol"], true_sol.repeat_interleave(s, dim=0))

    assert expanded["true_obj"].shape == (B * s,)
    assert torch.equal(expanded["true_obj"], true_obj.repeat_interleave(s, dim=0))

    assert isinstance(expanded["instance_kwargs"], dict)
    assert expanded["instance_kwargs"]["b"].shape == (B * s,)
    assert torch.equal(expanded["instance_kwargs"]["b"], instance_kwargs["b"].repeat_interleave(s, dim=0))

    assert expanded["instance_kwargs"]["c"].shape == (B * s, 3)
    assert torch.equal(expanded["instance_kwargs"]["c"], instance_kwargs["c"].repeat_interleave(s, dim=0))

    assert expanded["h"] == h
