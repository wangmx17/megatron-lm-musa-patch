"""Narrow training hooks that delegate to Megatron Core 0.19."""

from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Any

import megatron.training.training as training_module
from megatron.core.num_microbatches_calculator import get_num_microbatches

_ORIGINAL_TRAIN_STEP = training_module.train_step
_ORIGINAL_TRAINING_LOG = training_module.training_log
_ORIGINAL_TRAIN = training_module.train
_ACTIVE_PROFILER = ContextVar("musa_training_profiler", default=None)


def _fine_grained_offload_handler():
    """Return the legacy TE handler when that optional extension is available.

    TransformerEngine 2.17 no longer exposes this process-global handler;
    Megatron Core 0.19 owns fine-grained activation-offload scheduling instead.
    """

    try:
        from transformer_engine.pytorch import cpu_offload
    except ImportError:
        return None
    get_handler = getattr(cpu_offload, "get_fine_grained_offload_handler", None)
    return None if get_handler is None else get_handler()


def train_step(
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    opt_param_scheduler,
    config,
    forward_backward_func,
    iteration=None,
    pg_collection=None,
    p2p_communicator=None,
):
    """Run the complete 0.19 training step with optional legacy TE state."""

    handler = _fine_grained_offload_handler()
    if handler is not None:
        handler.num_microbatches = get_num_microbatches()
    result = _ORIGINAL_TRAIN_STEP(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
        iteration=iteration,
        pg_collection=pg_collection,
        p2p_communicator=p2p_communicator,
    )
    profiler = _ACTIVE_PROFILER.get()
    if profiler is not None:
        profiler.step()
    return result


def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate: float | None,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
    max_attention_logit,
    pg_collection=None,
    is_first_iteration=False,
    seqlen_squared_sum_in_batch: float | None = None,
    total_real_tokens_in_batch: float | None = None,
):
    """Preserve the 0.19 logger with the existing MUSA memory-report switch."""

    if os.environ.get("DISABLE_MEMORY_REPORT", "0") == "1":
        report_memory_flag = False
    return _ORIGINAL_TRAINING_LOG(
        loss_dict,
        total_loss_dict,
        learning_rate,
        iteration,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        grad_norm,
        params_norm,
        num_zeros_in_grad,
        max_attention_logit,
        pg_collection=pg_collection,
        is_first_iteration=is_first_iteration,
        seqlen_squared_sum_in_batch=seqlen_squared_sum_in_batch,
        total_real_tokens_in_batch=total_real_tokens_in_batch,
    )


def _register_debug_hooks(model: list[Any]) -> None:
    if os.environ.get("ENABLE_HOOK", "0") != "1":
        return
    from megatron.training.global_vars import get_args

    from musa_patch.debug_tools import DebugHookManager

    args = get_args()
    manager = DebugHookManager(
        enable_backward_hook=True,
        enable_forward_hook=True,
        enable_log_to_file=True,
        enable_log_online=False,
        valid_gradnorm_threads=10000,
        save_dir=args.tensorboard_dir,
    )
    manager.register_modulewise_hooks(model[0])


def train(
    forward_step_func,
    model,
    optimizer,
    opt_param_scheduler,
    train_data_iterator,
    valid_data_iterator,
    process_non_loss_data_func,
    config,
    checkpointing_context,
    non_loss_data_func,
    inference_model=None,
    p2p_communicator=None,
    pg_collection=None,
):
    """Initialize MUSA hooks, then run the complete 0.19 training loop."""

    handler = _fine_grained_offload_handler()
    if handler is not None:
        handler.init_by_config(config)
    _register_debug_hooks(model)
    from megatron.training.global_vars import get_args

    from .profiling import maybe_enable_profiling

    args = get_args()
    with maybe_enable_profiling(args, args.iteration) as profiler:
        token = _ACTIVE_PROFILER.set(profiler)
        try:
            return _ORIGINAL_TRAIN(
                forward_step_func,
                model,
                optimizer,
                opt_param_scheduler,
                train_data_iterator,
                valid_data_iterator,
                process_non_loss_data_func,
                config,
                checkpointing_context,
                non_loss_data_func,
                inference_model=inference_model,
                p2p_communicator=p2p_communicator,
                pg_collection=pg_collection,
            )
        finally:
            _ACTIVE_PROFILER.reset(token)


training_module.train_step = train_step
training_module.training_log = training_log
training_module.train = train
