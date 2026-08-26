"""Fix TE fused cross-entropy when SQ==1 leaves a stale stride(-2).

TE's online_softmax indexes X[row] = ptr + row * stride(-2) and assumes that
equals V. Megatron logits are often a [s, 1, V] view of [1, s, V], so
stride(-2) = s*V. PyTorch still reports is_contiguous()=True for a size-1 dim,
so contiguous()/clone(preserve_format) are no-ops. Rewrite that unused stride
(or copy if SQ>1) before calling TE.
"""
from __future__ import annotations

import torch
import transformer_engine.pytorch.triton.cross_entropy as tce


def _pack_for_te_kernel(t: torch.Tensor) -> torch.Tensor:
    _, sq, v = t.shape
    if t.stride(-1) == 1 and t.stride(-2) == v:
        return t
    if sq == 1 and t.stride(-1) == 1:
        return torch.as_strided(t, t.size(), (t.stride(0), v, 1))
    packed = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    packed.copy_(t)
    return packed


_orig_ce_fwd = tce.cross_entropy_forward


def _cross_entropy_forward(_input, *args, **kwargs):
    _input = _pack_for_te_kernel(_input)
    return _orig_ce_fwd(_input, *args, **kwargs)


tce.cross_entropy_forward = _cross_entropy_forward
