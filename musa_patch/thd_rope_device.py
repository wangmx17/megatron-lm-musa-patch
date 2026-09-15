"""Device-side metadata preparation for packed THD RoPE on MUSA."""

import ctypes
from functools import lru_cache
import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_DEVICE_METADATA_ENABLED = os.getenv("MUSA_THD_ROPE_DEVICE_METADATA", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


if triton is not None:

    @triton.jit
    def gather_cp_freqs(
        freqs,
        cu_seqlens,
        output,
        rotary_dim: tl.constexpr,
        num_sequences: tl.constexpr,
        cp_size: tl.constexpr,
        cp_rank: tl.constexpr,
        freq_seq_stride: tl.constexpr,
        freq_dim_stride: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Map each local packed token to its context-parallel RoPE frequency."""
        token = tl.program_id(0)
        lo = tl.full((), 0, tl.int32)
        hi = tl.full((), num_sequences, tl.int32)

        # Upper-bound search over local cumulative sequence ends. Empty spans
        # are handled by advancing until an end strictly greater than token.
        while lo < hi:
            mid = (lo + hi) // 2
            local_end = tl.load(cu_seqlens + mid + 1) // cp_size
            go_right = token >= local_end
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)

        local_start = tl.load(cu_seqlens + lo) // cp_size
        local_end = tl.load(cu_seqlens + lo + 1) // cp_size
        position = token - local_start

        if cp_size > 1:
            # Megatron CP lays out two symmetric chunks per rank. Reconstruct
            # the position in the full sequence without reading metadata on CPU.
            half_local_length = (local_end - local_start) // 2
            position = tl.where(
                position < half_local_length,
                cp_rank * half_local_length + position,
                (local_end - local_start) * cp_size
                - (cp_rank + 1) * half_local_length
                + position
                - half_local_length,
            )

        offsets = tl.arange(0, block_size)
        mask = offsets < rotary_dim
        values = tl.load(
            freqs + position * freq_seq_stride + offsets * freq_dim_stride,
            mask=mask,
            other=0.0,
        )
        tl.store(output + token * rotary_dim + offsets, values, mask=mask)


@lru_cache(maxsize=1)
def _load_musa_driver_symbols():
    """Keep MUSA driver symbols global for Triton's cached launcher."""
    try:
        return ctypes.CDLL("libmusa.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # Some installations already export the symbols globally or use a
        # different soname. Let Triton surface its native error if needed.
        return None


def device_thd_rope_supported(t, cu_seqlens, freqs):
    """Return whether the no-D2H THD RoPE fast path supports these tensors."""
    return (
        _DEVICE_METADATA_ENABLED
        and triton is not None
        and t.device.type == "musa"
        and cu_seqlens.device == t.device
        and freqs.device == t.device
        and t.ndim == 3
        and cu_seqlens.ndim == 1
        and cu_seqlens.dtype == torch.int32
        and cu_seqlens.is_contiguous()
        and cu_seqlens.numel() >= 2
        and t.shape[0] > 0
        and freqs.ndim == 4
        and tuple(freqs.shape[1:-1]) == (1, 1)
        # The native MUSA torch.rope kernel used here requires full rotary
        # dimensions. Preserve the existing path for partial rotary inputs.
        and freqs.shape[-1] == t.shape[-1]
        and freqs.shape[-1] % 2 == 0
        and not freqs.requires_grad
    )


def apply_rotary_pos_emb_thd_torch_rope_device(
    t,
    cu_seqlens,
    freqs,
    rotary_interleaved=False,
    cp_group=None,
):
    """Apply native MUSA RoPE after gathering packed THD frequencies on device.

    Sequence lengths must follow Megatron's THD contract: every sequence is
    evenly divisible by the context-parallel size and, for CP > 1, each local
    sequence contains the two symmetric chunks assigned to that rank. This
    fast path requires full-dimension RoPE.
    """
    if not device_thd_rope_supported(t, cu_seqlens, freqs):
        raise ValueError("unsupported tensors for MUSA THD RoPE device metadata fast path")

    cp_size = cp_group.size() if cp_group is not None else 1
    cp_rank = cp_group.rank() if cp_group is not None else 0
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"invalid context-parallel rank {cp_rank}/{cp_size}")

    _load_musa_driver_symbols()
    num_tokens = t.shape[0]
    rotary_dim = freqs.shape[-1]
    mapped_freqs = torch.empty(
        (num_tokens, rotary_dim), device=t.device, dtype=freqs.dtype
    )
    gather_cp_freqs[(num_tokens,)](
        freqs,
        cu_seqlens,
        mapped_freqs,
        rotary_dim,
        cu_seqlens.numel() - 1,
        cp_size,
        cp_rank,
        freqs.stride(0),
        freqs.stride(-1),
        triton.next_power_of_2(rotary_dim),
    )
    return torch.rope(
        t.unsqueeze(1), mapped_freqs, rotary_interleaved, False, False
    ).squeeze(1)


__all__ = [
    "apply_rotary_pos_emb_thd_torch_rope_device",
    "device_thd_rope_supported",
]
