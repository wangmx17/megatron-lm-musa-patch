"""Overlap routed FC1 weight gradients with backward ACE combine.

Transformer Engine still computes FC1 and FC2 wgrad and accumulates into the
original FP32 ``main_grad`` buffers. This adapter changes only when the two
native delayed-wgrad stores are drained: FC1 before the ACE completion wait,
then FC2 after it.
"""

from __future__ import annotations

from contextvars import ContextVar

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


_ACTIVE_WGRAD = ContextVar("deepep_ace_active_fc1_wgrad", default=None)
_ORIGINAL_FUSED_DISPATCH_FORWARD = FusedDispatch.forward
_ORIGINAL_DEEPEP_DISPATCH = _DeepepManager.dispatch
_ORIGINAL_MOE_INIT = MoELayer.__init__


class RoutedFc1Wgrad:
    """Own the native TE delayed-wgrad stores for one routed MoE layer."""

    def __init__(self, experts, config, groups):
        from transformer_engine.pytorch.module.grouped_linear import GroupedLinear

        try:
            from megatron.training import get_args

            overlap_grad_reduce = bool(getattr(get_args(), "overlap_grad_reduce", False))
        except (AssertionError, RuntimeError):
            overlap_grad_reduce = False

        requirements = {
            "shared-expert overlap disabled": not config.moe_shared_expert_overlap,
            "pipeline parallel size 1": config.pipeline_model_parallel_size == 1,
            "recompute disabled": config.recompute_granularity is None,
            "CPU offload disabled": not config.cpu_offloading,
            "FP8/FP4 disabled": not config.fp8 and not getattr(config, "fp4", None),
            "BF16 parameters": config.params_dtype == torch.bfloat16,
            "linear bias disabled": not config.add_bias_linear,
            "fused gradient accumulation": config.gradient_accumulation_fusion,
            "global delayed wgrad disabled": not config.delay_wgrad_compute,
            "asynchronous gradient reduction disabled": not overlap_grad_reduce,
            "flex token dispatcher": config.moe_token_dispatcher_type == "flex",
            "DeepEP flex backend": config.moe_flex_dispatcher_backend == "deepep",
            "expert tensor parallel size 1": (
                torch.distributed.get_world_size(groups.expt_tp) == 1
            ),
            "expert data parallel size 1": (
                torch.distributed.get_world_size(groups.expt_dp) == 1
            ),
        }
        failed_requirements = [
            name for name, satisfied in requirements.items() if not satisfied
        ]
        if failed_requirements:
            raise RuntimeError(
                "ACE FC1 wgrad overlap unsupported configuration: "
                + ", ".join(failed_requirements)
            )

        # Backward reaches FusedDispatch only after both expert GroupedLinear
        # stores have been populated. FC1 is deliberately drained first so it
        # overlaps ACE; FC2 remains after the completion wait.
        self.layers = (experts.linear_fc1, experts.linear_fc2)
        self.layer_names = ("fc1", "fc2")
        for layer in self.layers:
            if not isinstance(layer, GroupedLinear) or not layer.fuse_wgrad_accumulation:
                raise RuntimeError(
                    "ACE FC1 wgrad overlap requires the validated MUSA TE GroupedLinear"
                )
            if not all(param.requires_grad for param in layer.parameters()):
                raise RuntimeError("ACE FC1 wgrad overlap does not support frozen expert weights")
            store = type(layer.wgrad_store)(True)
            if not all(
                hasattr(store, name) for name in ("context", "put", "pop", "assert_empty")
            ):
                raise RuntimeError("Installed TE delayed-wgrad interface is incompatible")
            layer.wgrad_store = store
        self.native_backward_dw = GroupedLinear.backward_dw
        self.active = False
        self.next_layer = 0

    def begin(self):
        if (
            self.active
            or self.next_layer != 0
            or any(not layer.wgrad_store.context.empty() for layer in self.layers)
        ):
            raise RuntimeError("ACE FC1 wgrad overlap does not support in-flight reuse")
        self.active = True

    def _drain_until(self, end):
        if not self.active:
            raise RuntimeError("ACE FC1 wgrad drain requires an active backward")
        expected_sizes = (0,) * self.next_layer + (1,) * (
            len(self.layers) - self.next_layer
        )
        actual_sizes = tuple(layer.wgrad_store.context.qsize() for layer in self.layers)
        if actual_sizes != expected_sizes:
            raise RuntimeError("Expected exactly one routed FC1 and FC2 dW")
        while self.next_layer < end:
            name = self.layer_names[self.next_layer]
            layer = self.layers[self.next_layer]
            with torch.autograd.profiler.record_function("ACEWgrad::" + name):
                self.native_backward_dw(layer)
            layer.wgrad_store.assert_empty()
            self.next_layer += 1

    def drain_before_wait(self):
        self._drain_until(1)

    def drain_after_wait(self):
        self._drain_until(2)
        self.active = False
        self.next_layer = 0


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
            raise RuntimeError("ACE FC1 wgrad overlap requires async DeepEP completion")
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
        ctx.wgrad_overlap.drain_before_wait()
    if ctx.async_finish:
        after_event.current_stream_wait()
    if ctx.wgrad_overlap is not None:
        ctx.wgrad_overlap.drain_after_wait()
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
    token = _ACTIVE_WGRAD.set(getattr(self, "fc1_wgrad_overlap", None))
    try:
        return _ORIGINAL_DEEPEP_DISPATCH(
            self,
            hidden_states,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
    finally:
        _ACTIVE_WGRAD.reset(token)


def _set_fc1_wgrad_overlap(self, state):
    self._comm_manager.fc1_wgrad_overlap = state


def _moe_init(self, config, submodules=None, layer_number=None, pg_collection=None):
    groups = pg_collection if pg_collection is not None else get_default_pg_collection()
    _ORIGINAL_MOE_INIT(
        self,
        config,
        submodules=submodules,
        layer_number=layer_number,
        pg_collection=groups,
    )
    self.token_dispatcher.set_fc1_wgrad_overlap(
        RoutedFc1Wgrad(self.experts, config, groups)
    )


FusedDispatch.forward = staticmethod(_fused_dispatch_forward)
FusedDispatch.backward = staticmethod(_fused_dispatch_backward)
fused_a2a.fused_dispatch = _fused_dispatch
token_dispatcher.fused_dispatch = _fused_dispatch
_DeepepManager.dispatch = _deepep_dispatch
MoEFlexTokenDispatcher.set_fc1_wgrad_overlap = _set_fc1_wgrad_overlap
MoELayer.__init__ = _moe_init
