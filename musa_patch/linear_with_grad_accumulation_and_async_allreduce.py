"""MUSA communication-order guards for Megatron Core 0.19 tensor-parallel linear."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from megatron.core.tensor_parallel import layers


class _WaitedWork:
    """A Work-compatible proxy for an operation already synchronized on MUSA."""

    def __init__(self, work: Any, wait_result: Any):
        self._work = work
        self._wait_result = wait_result

    def wait(self, *_args: Any, **_kwargs: Any) -> Any:
        return self._wait_result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._work, name)


def _synchronize_async_collective(original: Callable[..., Any]):
    """Force completion where the previous MUSA autograd fork waited eagerly."""

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        work = original(*args, **kwargs)
        if kwargs.get("async_op", False):
            return _WaitedWork(work, work.wait())
        return work

    return wrapped


if not getattr(layers, "_musa_core_v019_linear_comm_patch", False):
    layers.dist_all_gather_func = _synchronize_async_collective(
        layers.dist_all_gather_func
    )
    layers.dist_reduce_scatter_func = _synchronize_async_collective(
        layers.dist_reduce_scatter_func
    )
    layers._musa_core_v019_linear_comm_patch = True
