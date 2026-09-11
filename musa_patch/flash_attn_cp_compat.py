"""MUSA flash-attention compatibility shim for Transformer-Engine CP/THD path.

Transformer-Engine's context-parallel attention (`AttnFuncWithCPAndKVP2P`) resolves a
*local* ``flash_attn_fwd`` at call time. For the flash-attn-2 branch it binds::

    flash_attn_fwd = _flash_attn_varlen_fwd          # == flash_attn_varlen_forward

and later reads ``fa_outputs[4]`` (out_padded), ``fa_outputs[5]`` (softmax_lse) and
``fa_outputs[7]`` (rng_state), i.e. it expects the dense-style 8-tuple layout::

    (out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state)

The MUSA build of ``_flash_attn_varlen_forward`` returns only a **4-tuple**::

    (out, softmax_lse, S_dmask, rng_state)

so ``fa_outputs[4]``/``fa_outputs[5]``/``fa_outputs[7]`` raise
``IndexError: tuple index out of range``.

This shim wraps ``flash_attn.flash_attn_interface._flash_attn_varlen_forward`` and
re-pads the 4-tuple into the 8-tuple TE expects (out_padded := out for MUSA, and the
q/k/v inputs echoed back, matching the dense ``_flash_attn_forward`` layout). It must be
installed **before** ``transformer_engine.pytorch.attention`` binds the name at import
time; ``musa_patch/__init__.py`` imports this module first in
``patch_before_import_megatron()`` for exactly that reason.
"""

import logging
import os
import sys

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the MUSA training image provides Triton
    triton = None
    tl = None

logger = logging.getLogger(__name__)


def get_te_thd_lse_fp32_mode():
    """Return the requested TE THD LSE fp32 mode.

    ``auto`` uses native fp32 when the installed TE advertises support,
    ``require`` additionally fails fast when that support is unavailable, and
    ``disable`` forces the established fp64 compatibility path for A/B tests.
    """
    mode = os.getenv("MUSA_TE_THD_LSE_FP32", "auto").strip().lower()
    if mode not in ("auto", "require", "disable"):
        raise RuntimeError(
            "MUSA_TE_THD_LSE_FP32 must be one of: auto, require, disable; "
            f"got {mode!r}"
        )
    return mode


if triton is not None:

    @triton.jit
    def _thd_second_half_lse_correction_fp32_kernel(
        lse_ptr,
        lse_per_step_ptr,
        half_seqlen: tl.constexpr,
        lse_seqlen: tl.constexpr,
        lse_per_step_seqlen: tl.constexpr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Merge the fp32 cast/correction/cast chain into one fp32 kernel."""
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        row = offsets // half_seqlen
        col = offsets - row * half_seqlen
        lse_offsets = row * lse_seqlen + half_seqlen + col
        step_offsets = row * lse_per_step_seqlen + col

        current = tl.load(lse_ptr + lse_offsets, mask=mask).to(tl.float32)
        update = tl.load(lse_per_step_ptr + step_offsets, mask=mask).to(tl.float32)
        max_value = tl.maximum(current, update)
        min_value = tl.minimum(current, update)
        corrected = max_value + tl.log(1.0 + tl.exp(min_value - max_value))
        tl.store(lse_ptr + lse_offsets, corrected, mask=mask)


def _can_use_thd_lse_fp32_fusion(lse, lse_per_step, cu_seqlens, lse_packed):
    """Return whether the current single-sequence THD layout fits the fused kernel."""
    del lse_packed  # packed and non-packed layouts are identical when batch == 1.
    if triton is None:
        return False
    if not (isinstance(lse, torch.Tensor) and isinstance(lse_per_step, torch.Tensor)):
        return False
    if lse.device.type != "musa" or lse_per_step.device != lse.device:
        return False
    if lse.dtype != torch.float32 or lse_per_step.dtype != torch.float32:
        return False
    if not (lse.is_contiguous() and lse_per_step.is_contiguous()):
        return False
    if lse.dim() not in (2, 3) or lse_per_step.dim() != lse.dim():
        return False
    if lse.dim() == 3 and (lse.size(0) != 1 or lse_per_step.size(0) != 1):
        return False
    if not isinstance(cu_seqlens, torch.Tensor):
        return False
    if cu_seqlens.device != lse.device or cu_seqlens.dtype != torch.int32:
        return False
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() != 2:
        return False
    if lse.shape[:-1] != lse_per_step.shape[:-1]:
        return False
    lse_seqlen = lse.size(-1)
    return lse_seqlen > 0 and lse_seqlen % 2 == 0 and lse_per_step.size(-1) >= lse_seqlen // 2


def _run_thd_lse_fp32_fusion(lse, lse_per_step):
    lse_seqlen = lse.size(-1)
    half_seqlen = lse_seqlen // 2
    rows = lse.numel() // lse_seqlen
    n_elements = rows * half_seqlen
    block_size = 256
    _thd_second_half_lse_correction_fp32_kernel[
        (triton.cdiv(n_elements, block_size),)
    ](
        lse,
        lse_per_step,
        half_seqlen,
        lse_seqlen,
        lse_per_step.size(-1),
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


def install_te_thd_aux_fusion(te_attention_module):
    """Install the TE THD LSE compatibility path and optional fused fastpath.

    A Transformer Engine build advertising native fp32 support is already the
    preferred path and needs no wrapper.  Older MUSA builds require a float64
    aggregate LSE while FlashAttention produces float32, so their compatibility
    path launches fp32->fp64, correction, fp64->fp32, and copy kernels on every CP
    step. ``MUSA_FA_AUX_FUSION=1`` replaces that chain for the current
    single-sequence THD shape with one in-place fp32 Triton kernel. Every
    unsupported layout keeps the established conversion path.
    """
    tex = getattr(te_attention_module, "tex", None)
    original = getattr(tex, "thd_second_half_lse_correction", None)
    if original is None or getattr(original, "_musa_aux_fusion_wrapper", False):
        return
    mode = get_te_thd_lse_fp32_mode()
    has_native_fp32 = bool(getattr(tex, "NVTE_MUSA_THD_LSE_FP32", False))
    if mode == "require" and not has_native_fp32:
        raise RuntimeError(
            "MUSA_TE_THD_LSE_FP32=require, but Transformer Engine does not "
            "advertise native THD LSE fp32 support"
        )
    if has_native_fp32 and mode != "disable":
        logger.info(
            "[musa_patch] Transformer Engine provides native THD LSE fp32 support"
        )
        return

    enabled = os.getenv("MUSA_FA_AUX_FUSION", "0") == "1"
    first_hit = [True]

    def _thd_second_half_lse_correction(lse, lse_per_step, cu_seqlens, lse_packed):
        if enabled and _can_use_thd_lse_fp32_fusion(
            lse, lse_per_step, cu_seqlens, lse_packed
        ):
            _run_thd_lse_fp32_fusion(lse, lse_per_step)
            if first_hit[0]:
                logger.warning(
                    "[musa_patch] FlashAttention THD auxiliary fp32 fusion reached "
                    "for shape=%s, per_step_shape=%s",
                    tuple(lse.shape),
                    tuple(lse_per_step.shape),
                )
                first_hit[0] = False
            return None

        if lse is not None and lse.dtype != torch.float64:
            lse_double = lse.to(torch.float64)
            out = original(lse_double, lse_per_step, cu_seqlens, lse_packed)
            lse.copy_(lse_double.to(dtype=lse.dtype))
            return out
        return original(lse, lse_per_step, cu_seqlens, lse_packed)

    _thd_second_half_lse_correction._musa_aux_fusion_wrapper = True
    tex.thd_second_half_lse_correction = _thd_second_half_lse_correction
    logger.info(
        "[musa_patch] FlashAttention THD auxiliary fusion installed (enabled=%s)", enabled
    )


def _maybe_install_te_thd_aux_fusion():
    """Install lazily after TE has bound the patched FlashAttention entry point."""
    if os.getenv("MUSA_FA_AUX_FUSION", "0") != "1":
        return
    te_attention = sys.modules.get("transformer_engine.pytorch.attention")
    if te_attention is not None:
        install_te_thd_aux_fusion(te_attention)


def _install_flash_attn_cp_compat():
    try:
        import flash_attn.flash_attn_interface as _fai
    except ImportError:  # pragma: no cover - flash-attn not present
        logger.warning("[musa_patch] flash_attn not installed; CP/THD compat shim skipped.")
        return

    _orig_varlen_forward = _fai._flash_attn_varlen_forward

    def _patched_flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        *args,
        **kwargs,
    ):
        _maybe_install_te_thd_aux_fusion()
        out, softmax_lse, S_dmask, rng_state = _orig_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            *args,
            **kwargs,
        )
        # Keep the lse in MUSA's native 3-D fp32 form [num_seq, nheads, total_t]:
        # the MUSA varlen *backward* kernel and TE's `lse_per_step` both expect Float, and
        # TE's aggregate lse (dim==2, Double) is derived from it locally in TE's forward.
        if softmax_lse is not None and softmax_lse.dtype != torch.float32:
            softmax_lse = softmax_lse.to(torch.float32)
        # Re-pad MUSA's 4-tuple into TE's expected dense-style 8-tuple layout.
        # (out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state)
        out_padded = out
        return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state

    _fai._flash_attn_varlen_forward = _patched_flash_attn_varlen_forward

    # TE 2.5 path (musa_patch pins _flash_attn_2_6_0_plus=False) does not pass
    # softcap; MUSA _flash_attn_varlen_backward requires it as a positional.
    _orig_varlen_backward = getattr(_fai, "_flash_attn_varlen_backward", None)

    def _patched_flash_attn_varlen_backward(*args, **kwargs):
        _maybe_install_te_thd_aux_fusion()
        kwargs.setdefault("softcap", 0.0)
        return _orig_varlen_backward(*args, **kwargs)

    if _orig_varlen_backward is not None:
        _fai._flash_attn_varlen_backward = _patched_flash_attn_varlen_backward

    # TE binds the name under several aliases at import time; cover every one it may pick.
    for _alias in (
        "_flash_attn_varlen_fwd",
        "flash_attn_varlen_fwd",
    ):
        if getattr(_fai, _alias, None) is not None:
            setattr(_fai, _alias, _patched_flash_attn_varlen_forward)
    for _alias in (
        "_flash_attn_varlen_bwd",
        "flash_attn_varlen_bwd",
    ):
        if getattr(_fai, _alias, None) is not None and _orig_varlen_backward is not None:
            setattr(_fai, _alias, _patched_flash_attn_varlen_backward)

    logger.info(
        "[musa_patch] installed flash-attn CP/THD compat wrapper on "
        "_flash_attn_varlen_forward/_flash_attn_varlen_backward"
    )


_install_flash_attn_cp_compat()
