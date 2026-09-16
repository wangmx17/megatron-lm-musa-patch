"""DeepEP ACE integration for Megatron's fused all-to-all path.

``token_num`` in the ACE constructor is the *local dispatch input capacity*.
The native combine capacity is ``token_num * num_topk``.  Deriving it from the
first FusedDispatch input avoids the multi-GiB over-allocation caused by using
the EP-global or top-k-expanded token count.
"""

from __future__ import annotations

import importlib.metadata
import os

import torch
import megatron.core.transformer.moe.fused_a2a as fused_a2a
from megatron.core.transformer.moe.fused_a2a import Buffer, FusedDispatch


_BUFFER_LOGGED = False
_COMBINE_LOGGED = False
_BACKWARD_COMBINE_CACHE_LOGGED = False
_DISPATCH_CACHE_LOGGED = False
_RUNTIME_LOCAL_TOKEN_CAPACITY = None
_RUNTIME_HIDDEN_SIZE = None
_RUNTIME_TOPK = None
_RUNTIME_TOKEN_SOURCE = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


_ORIGINAL_FUSED_DISPATCH_FORWARD = FusedDispatch.forward


def _fused_dispatch_forward(
    ctx,
    x,
    token_indices,
    token_probs,
    num_experts,
    group,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    """Capture the real local input geometry before get_buffer is called."""
    global _RUNTIME_LOCAL_TOKEN_CAPACITY, _RUNTIME_HIDDEN_SIZE, _RUNTIME_TOPK
    global _RUNTIME_TOKEN_SOURCE
    global _DISPATCH_CACHE_LOGGED

    local_tokens = int(x.size(0))
    hidden_size = int(x.size(1))
    actual_topk = int(token_indices.size(1))
    dynamic_tokens = _env_int("DEEPEP_ACE_DYNAMIC_TOKEN_NUM", 1)
    if dynamic_tokens not in (0, 1):
        raise ValueError(
            "DEEPEP_ACE_DYNAMIC_TOKEN_NUM must be 0 or 1, "
            f"got {dynamic_tokens}"
        )
    # The legacy launcher defaults DEEPEP_ACE_TOKEN_NUM to the EP-global
    # sequence count (65536).  That is not this native API's semantic and
    # over-allocates several GiB.  Dynamic mode is therefore the safe default.
    configured_tokens = (
        0 if dynamic_tokens else _env_int("DEEPEP_ACE_TOKEN_NUM", 0)
    )
    configured_hidden = _env_int("DEEPEP_ACE_HIDDEN", 0)
    configured_topk = _env_int("DEEPEP_ACE_NUM_TOPK", 0)
    if configured_tokens and configured_tokens < local_tokens:
        raise RuntimeError(
            "DEEPEP_ACE_TOKEN_NUM is a local dispatch capacity and is too small: "
            f"configured={configured_tokens}, input_rows={local_tokens}"
        )
    if configured_hidden and configured_hidden != hidden_size:
        raise RuntimeError(
            f"DEEPEP_ACE_HIDDEN={configured_hidden} does not match x.size(1)={hidden_size}"
        )
    if configured_topk and configured_topk != actual_topk:
        raise RuntimeError(
            f"DEEPEP_ACE_NUM_TOPK={configured_topk} does not match routing topk={actual_topk}"
        )
    _RUNTIME_LOCAL_TOKEN_CAPACITY = configured_tokens or local_tokens
    _RUNTIME_HIDDEN_SIZE = configured_hidden or hidden_size
    _RUNTIME_TOPK = configured_topk or actual_topk
    _RUNTIME_TOKEN_SOURCE = "configured-local" if configured_tokens else "dispatch-x"
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


FusedDispatch.forward = staticmethod(_fused_dispatch_forward)


def get_buffer(group, hidden_bytes: int):
    """Create the shared DeepEP buffer with correctly sized ACE metadata."""
    global _BUFFER_LOGGED

    if _RUNTIME_LOCAL_TOKEN_CAPACITY is None:
        raise RuntimeError(
            "ACE buffer requested before observing FusedDispatch input geometry; "
            "set DEEPEP_ACE_TOKEN_NUM, DEEPEP_ACE_HIDDEN and DEEPEP_ACE_NUM_TOPK explicitly"
        )
    num_nvl_bytes = 0
    num_rdma_bytes = 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            num_nvl_bytes,
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
        )
        num_rdma_bytes = max(
            num_rdma_bytes,
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
        )

    token_num = _RUNTIME_LOCAL_TOKEN_CAPACITY
    hidden_size = _RUNTIME_HIDDEN_SIZE
    num_topk = _RUNTIME_TOPK
    num_ace_buffers = max(_env_int("DEEPEP_ACE_NUM_BUFFERS", 1), 1)
    buffer = fused_a2a._buffer
    need_new = (
        buffer is None
        or buffer.group != group
        or buffer.num_nvl_bytes < num_nvl_bytes
        or buffer.num_rdma_bytes < num_rdma_bytes
        or not getattr(buffer, "use_ace", False)
        or getattr(buffer, "_musa_ace_token_num", 0) < token_num
        or getattr(buffer, "_musa_ace_hidden_size", 0) != hidden_size
        or getattr(buffer, "_musa_ace_num_topk", 0) != num_topk
        or getattr(buffer, "_musa_ace_num_buffers", 0) != num_ace_buffers
    )
    if need_new:
        fused_a2a._buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            use_ace=True,
            num_ace_buffers=num_ace_buffers,
            token_num=token_num,
            hidden_size=hidden_size,
            num_topk=num_topk,
        )
        buffer = fused_a2a._buffer
        buffer._musa_ace_token_num = token_num
        buffer._musa_ace_hidden_size = hidden_size
        buffer._musa_ace_num_topk = num_topk
        buffer._musa_ace_num_buffers = num_ace_buffers

    if not hasattr(buffer, "get_ace_combine_buffer"):
        raise RuntimeError("The installed DeepEP does not expose ACE combine buffers")
    if not _BUFFER_LOGGED and group.rank() == 0:
        _BUFFER_LOGGED = True
        try:
            version = importlib.metadata.version("deep-ep")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        print(
            "[deepep_ace] Buffer ready "
            f"version={version} local_token_capacity={token_num} "
            f"token_source={_RUNTIME_TOKEN_SOURCE} "
            f"combine_capacity={token_num * num_topk} hidden={hidden_size} "
            f"topk={num_topk} buffers={num_ace_buffers} group_size={group.size()} "
            f"nvl_bytes={num_nvl_bytes}",
            flush=True,
        )
    return buffer


_ORIGINAL_BUFFER_COMBINE = Buffer.combine


def _validated_buffer_combine(
    self,
    x,
    handle,
    topk_weights=None,
    bias=None,
    config=None,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
):
    """Validate that every ACE combine input is backed by an ACE window."""
    global _COMBINE_LOGGED, _BACKWARD_COMBINE_CACHE_LOGGED

    if getattr(self, "use_ace", False):
        capacity = getattr(self, "_musa_ace_token_num", 0) * getattr(
            self, "_musa_ace_num_topk", 0
        )
        if x.size(0) > capacity:
            raise RuntimeError(
                f"ACE combine rows={x.size(0)} exceed capacity={capacity}; "
                "increase the local DEEPEP_ACE_TOKEN_NUM"
            )
        with_topk = topk_weights is not None
        num_topk = topk_weights.size(1) if with_topk else 1
        buffer_index = _env_int("DEEPEP_ACE_BUFFER_INDEX", 0)
        expected_x, expected_probs = self.get_ace_combine_buffer(
            x.size(0), x.size(1), num_topk, with_topk, buffer_index
        )
        if x.data_ptr() != expected_x.data_ptr():
            raise RuntimeError(
                "ACE combine input is not backed by get_ace_combine_buffer(); "
                "enable the MUSA DeepEP token-dispatcher preallocation patch"
            )
        if with_topk and topk_weights.data_ptr() != expected_probs.data_ptr():
            raise RuntimeError(
                "ACE combine top-k weights are not backed by get_ace_combine_buffer()"
            )
        if not _COMBINE_LOGGED and self.rank == 0:
            _COMBINE_LOGGED = True
            print(
                "[deepep_ace] registered combine input verified "
                f"shape={tuple(x.shape)} with_topk={with_topk} topk={num_topk}",
                flush=True,
            )

    return _ORIGINAL_BUFFER_COMBINE(
        self,
        x,
        handle,
        topk_weights=topk_weights,
        bias=bias,
        config=config,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=allocate_on_comm_stream,
    )


fused_a2a.get_buffer = get_buffer
Buffer.combine = _validated_buffer_combine
