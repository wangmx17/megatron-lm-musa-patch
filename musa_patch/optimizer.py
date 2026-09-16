"""Narrow MUSA optimizer compatibility hooks for Megatron Core 0.19."""

from __future__ import annotations

import os
import warnings
from typing import Any

import megatron.core.optimizer as optimizer_module
import torch
from packaging.version import Version

_ORIGINAL_GET_OPTIMIZER = getattr(
    optimizer_module,
    "_musa_original_get_megatron_optimizer_based_on_param_groups",
    optimizer_module._get_megatron_optimizer_based_on_param_groups,
)
optimizer_module._musa_original_get_megatron_optimizer_based_on_param_groups = (
    _ORIGINAL_GET_OPTIMIZER
)


def _legacy_deepspeed_cpu_adam():
    """Load the pre-PyTorch-2.3 CPU optimizer without CUDA probing loops."""

    original_is_available = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        return DeepSpeedCPUAdam
    except ImportError:
        warnings.warn(
            "DeepSpeedCPUAdam is unavailable; using Megatron's standard CPUAdam.",
            stacklevel=2,
        )
        return None
    finally:
        torch.cuda.is_available = original_is_available


def _get_megatron_optimizer_based_on_param_groups(config, *args: Any, **kwargs: Any):
    """Preserve 0.19 optimizer features while applying two MUSA CPU-offload guards."""

    if (
        config.optimizer_cpu_offload
        and os.environ.get("CPU_OPTIMIZER_PRECISION_AWARE_RECONFIG", "0") != "1"
    ):
        config.use_precision_aware_optimizer = False

    original_cpu_adam = optimizer_module.CPUAdam
    legacy_cpu_adam = None
    if config.optimizer_cpu_offload and Version(
        torch.__version__.split("+")[0]
    ) < Version("2.3.0"):
        legacy_cpu_adam = _legacy_deepspeed_cpu_adam()
        if legacy_cpu_adam is not None:
            optimizer_module.CPUAdam = legacy_cpu_adam
            warnings.warn(
                "Using DeepSpeedCPUAdam for MUSA CPU offload on PyTorch < 2.3.",
                stacklevel=2,
            )

    try:
        return _ORIGINAL_GET_OPTIMIZER(config, *args, **kwargs)
    finally:
        optimizer_module.CPUAdam = original_cpu_adam


optimizer_module._get_megatron_optimizer_based_on_param_groups = (
    _get_megatron_optimizer_based_on_param_groups
)
