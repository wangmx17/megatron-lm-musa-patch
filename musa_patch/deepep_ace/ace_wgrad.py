"""Overlap routed-expert weight gradients with the backward ACE combine.

This is a bounded adapter for the validated MiniCPM5 PP1 configuration.  It
uses Transformer Engine's native delayed-wgrad store only for routed FC2/FC1;
Megatron's global ``delay_wgrad_compute`` switch remains disabled.
"""

from __future__ import annotations

from contextvars import ContextVar
import os

import torch

import megatron.core.transformer.moe.fused_a2a as fused_a2a
import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
from megatron.core.transformer.moe.fused_a2a import (
    EventHandle,
    EventOverlap,
    FusedDispatch,
    get_buffer,
    get_hidden_bytes,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, get_default_pg_collection
from megatron.core.transformer.moe.token_dispatcher import (
    MoEFlexTokenDispatcher,
    _DeepepManager,
)


_ACTIVE_WGRAD = ContextVar("deepep_ace_active_wgrad", default=None)
_ORIGINAL_FUSED_DISPATCH_FORWARD = FusedDispatch.forward
_ORIGINAL_DEEPEP_DISPATCH = _DeepepManager.dispatch
_ORIGINAL_MOE_INIT = MoELayer.__init__


class RoutedWgrad:
    """Own the two native TE delayed-wgrad stores for one routed MoE layer."""

    def __init__(self, experts, config, groups):
        from transformer_engine.pytorch.module.grouped_linear import GroupedLinear

        try:
            from megatron.training import get_args

            overlap_grad_reduce = bool(getattr(get_args(), "overlap_grad_reduce", False))
        except (AssertionError, RuntimeError):
            overlap_grad_reduce = False

        conditions = (
            os.getenv("USE_DEEPEP_ACE") == "1",
            not config.moe_shared_expert_overlap,
            os.getenv("ENABLE_SHARED_AG_DISPATCH_OVERLAP", "0") == "0",
            config.pipeline_model_parallel_size == 1,
            config.recompute_granularity is None,
            not config.cpu_offloading,
            not config.fp8 and not getattr(config, "fp4", None),
            config.params_dtype == torch.bfloat16,
            not config.add_bias_linear,
            config.gradient_accumulation_fusion,
            not config.delay_wgrad_compute,
            not overlap_grad_reduce,
            config.moe_token_dispatcher_type == "flex",
            config.moe_flex_dispatcher_backend == "deepep",
            torch.distributed.get_world_size(groups.expt_tp) == 1,
            torch.distributed.get_world_size(groups.expt_dp) == 1,
        )
        if not all(conditions):
            raise RuntimeError(
                "ACE wgrad requires ACE-only, PP1, BF16, no recompute/offload/bias, "
                "no async grad reduction, fused accumulation, and expert TP/DP1"
            )
        self.layers = (experts.linear_fc2, experts.linear_fc1)
        for layer in self.layers:
            if not isinstance(layer, GroupedLinear) or not layer.fuse_wgrad_accumulation:
                raise RuntimeError("ACE wgrad requires the validated MUSA TE GroupedLinear")
            if not all(param.requires_grad for param in layer.parameters()):
                raise RuntimeError("ACE wgrad does not support frozen routed-expert weights")
            store = type(layer.wgrad_store)(True)
            if not all(
                hasattr(store, name) for name in ("context", "put", "pop", "assert_empty")
            ):
                raise RuntimeError("Installed TE delayed-wgrad interface is incompatible")
            layer.wgrad_store = store
        self.native_backward_dw = GroupedLinear.backward_dw
        self.active = False
        self.completed = 0

    def begin(self):
        if self.active or any(not layer.wgrad_store.context.empty() for layer in self.layers):
            raise RuntimeError("ACE wgrad does not support multiple in-flight forwards")
        self.active = True

    def drain(self):
        if not self.active or any(layer.wgrad_store.context.qsize() != 1 for layer in self.layers):
            raise RuntimeError("Expected exactly one routed FC2 and FC1 dW for this backward")
        for name, layer in zip(("fc2", "fc1"), self.layers):
            with torch.autograd.profiler.record_function("ACEWgrad::" + name):
                self.native_backward_dw(layer)
            layer.wgrad_store.assert_empty()
        self.active = False
        self.completed += 1


def _fused_dispatch_forward(
    ctx,
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    async_finish=False,
    allocate_on_comm_stream=False,
    wgrad_overlap=None,
):
    ctx.wgrad_overlap = wgrad_overlap
    if wgrad_overlap is not None:
        if not async_finish:
            raise RuntimeError("ACE wgrad requires async DeepEP completion")
        wgrad_overlap.begin()
    return _ORIGINAL_FUSED_DISPATCH_FORWARD(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish,
        allocate_on_comm_stream,
    )


def _fused_dispatch_backward(
    ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
):
    buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
    previous_event = EventOverlap(EventHandle()) if ctx.async_finish else None
    grad_x, grad_token_probs, after_event = buffer.combine(
        grad_output.contiguous(),
        ctx.handle,
        topk_weights=grad_token_probs.float(),
        previous_event=previous_event,
        async_finish=ctx.async_finish,
        allocate_on_comm_stream=ctx.allocate_on_comm_stream,
    )
    if ctx.wgrad_overlap is not None:
        ctx.wgrad_overlap.drain()
    if ctx.async_finish:
        after_event.current_stream_wait()
    result = (grad_x, None, grad_token_probs, None, None, None, None, None)
    return result[: len(ctx.needs_input_grad)]


def _fused_dispatch(
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    return FusedDispatch.apply(
        x.contiguous(),
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish,
        allocate_on_comm_stream,
        _ACTIVE_WGRAD.get() if torch.is_grad_enabled() else None,
    )


def _deepep_dispatch(self, hidden_states, async_finish=False, allocate_on_comm_stream=False):
    token = _ACTIVE_WGRAD.set(getattr(self, "wgrad_overlap", None))
    try:
        return _ORIGINAL_DEEPEP_DISPATCH(
            self,
            hidden_states,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    finally:
        _ACTIVE_WGRAD.reset(token)


def _set_routed_wgrad_overlap(self, state):
    self._comm_manager.wgrad_overlap = state


def _moe_init(self, config, submodules=None, layer_number=None, pg_collection=None):
    groups = pg_collection if pg_collection is not None else get_default_pg_collection()
    _ORIGINAL_MOE_INIT(
        self,
        config,
        submodules=submodules,
        layer_number=layer_number,
        pg_collection=groups,
    )
    enabled = os.getenv("ENABLE_ACE_WGRAD_OVERLAP", "0")
    if enabled not in ("0", "1"):
        raise ValueError("ENABLE_ACE_WGRAD_OVERLAP must be 0 or 1")
    if enabled == "1":
        self.token_dispatcher.set_routed_wgrad_overlap(RoutedWgrad(self.experts, config, groups))


FusedDispatch.forward = staticmethod(_fused_dispatch_forward)
FusedDispatch.backward = staticmethod(_fused_dispatch_backward)
fused_a2a.fused_dispatch = _fused_dispatch
token_dispatcher.fused_dispatch = _fused_dispatch
_DeepepManager.dispatch = _deepep_dispatch
MoEFlexTokenDispatcher.set_routed_wgrad_overlap = _set_routed_wgrad_overlap
MoELayer.__init__ = _moe_init
