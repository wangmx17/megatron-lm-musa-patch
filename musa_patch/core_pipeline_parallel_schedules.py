"""Profiler-only wrappers that preserve Megatron Core schedule signatures."""

from __future__ import annotations

import os
from typing import Any

from megatron.core.pipeline_parallel import schedules

from .utils import record_function_decorator

_ORIGINAL_FORWARD_STEP = schedules.forward_step
_ORIGINAL_BACKWARD_STEP = schedules.backward_step


@record_function_decorator
def forward_step(*args: Any, **kwargs: Any):
    return _ORIGINAL_FORWARD_STEP(*args, **kwargs)


@record_function_decorator
def backward_step(*args: Any, **kwargs: Any):
    return _ORIGINAL_BACKWARD_STEP(*args, **kwargs)


if os.environ.get("ENABLE_PROFILER", "0") == "1":
    schedules.forward_step = forward_step
    schedules.backward_step = backward_step
