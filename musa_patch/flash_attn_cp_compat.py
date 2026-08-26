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

import torch

logger = logging.getLogger(__name__)


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
