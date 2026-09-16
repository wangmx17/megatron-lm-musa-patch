"""Allocate Megatron/TE permutation outputs from DeepEP ACE windows."""

from __future__ import annotations

import torch

from megatron.core.extensions.transformer_engine import (
    fused_permute_with_probs,
    fused_unpermute,
)
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.transformer.moe.fused_a2a import get_buffer, get_hidden_bytes
from megatron.core.transformer.moe.token_dispatcher import _DeepepManager
from transformer_engine.musa.pytorch.utils import replace_attr


_FORWARD_LOGGED = False
_BACKWARD_LOGGED = False
_PROB_HOOK_LOGGED = False


def _get_ace_buffers(
    manager,
    hidden_states,
    *,
    num_tokens: int,
    hidden_size: int,
    num_topk: int,
    with_topk: bool,
):
    """Return views from the registered ACE window without a torch allocation."""
    buffer = get_buffer(manager.group, get_hidden_bytes(hidden_states))
    capacity = buffer._musa_ace_token_num * buffer._musa_ace_num_topk
    if num_tokens > capacity:
        raise RuntimeError(
            f"ACE view rows={num_tokens} exceed combine capacity={capacity} "
            f"({buffer._musa_ace_token_num} local tokens * "
            f"topk {buffer._musa_ace_num_topk})"
        )
    return buffer.get_ace_combine_buffer(
        num_tokens,
        hidden_size,
        num_topk,
        with_topk,
        0,
    )


def _route_prob_grad_to_ace(grad: torch.Tensor, ace_probs: torch.Tensor):
    """Copy index-form probability gradients into DeepEP's ACE window."""
    global _PROB_HOOK_LOGGED

    if grad is None:
        return None
    if grad.shape != ace_probs.shape:
        raise RuntimeError(
            "ACE routing-gradient shape mismatch: "
            f"grad={tuple(grad.shape)} buffer={tuple(ace_probs.shape)}"
        )
    if grad.dtype != ace_probs.dtype:
        raise RuntimeError(
            "ACE routing-gradient dtype mismatch: "
            f"grad={grad.dtype} buffer={ace_probs.dtype}"
        )
    if grad.data_ptr() != ace_probs.data_ptr():
        ace_probs.copy_(grad)
        grad = ace_probs
    if not _PROB_HOOK_LOGGED:
        _PROB_HOOK_LOGGED = True
        print(
            "[deepep_ace] routing probability gradient moved to registered buffer "
            f"shape={tuple(grad.shape)}",
            flush=True,
        )
    return grad


def _get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor):
    """Permute experts and preallocate its backward outputs for ACE combine."""
    global _BACKWARD_LOGGED

    dispatched_probs_indices = self.dispatched_probs
    self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
        self.dispatched_indices,
        dispatched_probs_indices,
        self.num_local_experts,
    )
    if self.config.moe_router_padding_for_quantization:
        self.dispatched_routing_map, self.tokens_per_expert = self._pad_routing_map(
            self.dispatched_routing_map,
            self.tokens_per_expert,
        )

    self.hidden_shape_before_permute = hidden_states.shape
    if self.dispatched_probs.dtype != torch.float32:
        raise TypeError(
            f"DeepEP ACE requires float32 routing probabilities, got {self.dispatched_probs.dtype}"
        )
    ace_hidden_states, ace_probs = _get_ace_buffers(
        self,
        hidden_states,
        num_tokens=hidden_states.size(0),
        hidden_size=hidden_states.size(1),
        num_topk=self.router_topk,
        with_topk=True,
    )
    if dispatched_probs_indices.requires_grad:
        dispatched_probs_indices.register_hook(
            lambda grad, target=ace_probs: _route_prob_grad_to_ace(grad, target)
        )
    hidden_states, permuted_probs, self.reversed_mapping_for_combine = (
        fused_permute_with_probs(
            hidden_states,
            self.dispatched_probs,
            self.dispatched_routing_map,
            num_out_tokens=self.tokens_per_expert.sum().item(),
            preallocated_act_b=ace_hidden_states,
        )
    )
    if self.router_dtype == "fp64":
        permuted_probs = permuted_probs.to(torch.float64)
    if not _BACKWARD_LOGGED and self.group.rank() == 0:
        _BACKWARD_LOGGED = True
        print(
            "[deepep_ace] TE permute hidden-gradient target registered "
            f"act_shape={tuple(ace_hidden_states.shape)} "
            f"index_prob_shape={tuple(ace_probs.shape)}",
            flush=True,
        )
    return hidden_states, permuted_probs


def _get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor):
    """Unpermute expert output directly into the ACE forward-combine input."""
    global _FORWARD_LOGGED

    num_tokens, hidden_size = self.hidden_shape_before_permute
    ace_hidden_states, _ = _get_ace_buffers(
        self,
        hidden_states,
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_topk=1,
        with_topk=False,
    )
    hidden_states = fused_unpermute(
        hidden_states,
        self.reversed_mapping_for_combine,
        restore_shape=self.hidden_shape_before_permute,
        preallocated_act_f=ace_hidden_states,
    )
    if hidden_states.data_ptr() != ace_hidden_states.data_ptr():
        raise RuntimeError("TE fused_unpermute ignored the ACE preallocated output buffer")
    if not _FORWARD_LOGGED and self.group.rank() == 0:
        _FORWARD_LOGGED = True
        print(
            "[deepep_ace] TE unpermute wrote registered combine input "
            f"shape={tuple(hidden_states.shape)}",
            flush=True,
        )
    return hidden_states


replace_attr(
    _DeepepManager,
    "get_permuted_hidden_states_by_experts",
    _get_permuted_hidden_states_by_experts,
)
replace_attr(
    _DeepepManager,
    "get_restored_hidden_states_by_experts",
    _get_restored_hidden_states_by_experts,
)
