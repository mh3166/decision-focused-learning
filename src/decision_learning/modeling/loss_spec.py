from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Any


@dataclass
class LossSpec:
    """Configuration for a loss experiment.

    Where does data go?
    - init_kwargs: fixed constructor kwargs for the loss.
    - hyper_grid: list of dicts (already expanded) to sweep over.
    - aux: experiment-level fixed objects (optmodel, minimize, reference modules).
      Do not put per-sample tensors here.
    - extra_batch_data: per-sample arrays/tensors keyed by name. These are split
      into train/val/test and merged into each batch dict before forward().

    Example:
        LossSpec(
            name="CustomLoss",
            factory=CustomLoss,
            init_kwargs={"alpha": 0.5},
            extra_batch_data={"extra": extra_tensor},
        )
    """
    name: str
    factory: Callable
    init_kwargs: dict
    hyper_grid: Iterable[dict] | None = None
    aux: dict = field(default_factory=dict)
    extra_batch_data: Mapping[str, Any] | None = None
