"""Back the ``flash_attn`` (FA2) API with MATE FlashAttention-3 kernels.

The MUSA training images used by this stack until now shipped a MUSA build of
flash-attn 2.x.  ``sense-time_veomni-rc0`` drops it and ships MATE FA3
(``flash_attn_3`` + ``mate``) instead, so both Transformer Engine and
``musa_patch.flash_attn_cp_compat`` fail at import: TE reads the ``flash-attn``
distribution metadata unconditionally, and the CP/THD shim wraps
``flash_attn.flash_attn_interface``.

This module recreates just enough of the FA2 surface on top of MATE:

* a synthetic ``flash-attn`` distribution so ``importlib.metadata.version``
  resolves to a release TE accepts (TE MUSA requires 2.0.6 <= v <= 2.6.8, and
  ``musa_patch`` pins its feature flags to the 2.5.0 behaviour);
* a ``flash_attn.flash_attn_interface`` module whose varlen entry points call
  MATE's low-level forward and its tilelang varlen backward.

Layouts follow what the previous MUSA FA2 build produced, because the TE CP
correction kernels (``tex.thd_*``) and ``flash_attn_cp_compat`` are already
validated against it: the varlen forward returns the MUSA 4-tuple ``(out,
softmax_lse, S_dmask, rng_state)`` and the log-sum-exp keeps the non-packed
``[batch, heads, total_tokens]`` shape (MATE returns ``[heads, total_tokens]``).

Set ``USE_MATE_FA3=0`` to skip the shim when a real flash-attn is installed.
"""

import logging
import os
import sys
import types
from importlib.metadata import Distribution, DistributionFinder

import torch

logger = logging.getLogger(__name__)

# TE MUSA accepts [2.0.6, 2.6.8]; musa_patch pins the 2.5.0 feature flags, which
# keeps softmax_lse in the non-packed layout the tex.thd_* kernels expect.
_FA2_COMPAT_VERSION = "2.5.0"

_INSTALLED = False


def mate_fa3_requested():
    """Return whether the MATE FA3 compatibility layer should be installed."""
    flag = os.getenv("USE_MATE_FA3", "auto").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return False
    if flag in ("1", "true", "on", "yes"):
        return True
    if flag != "auto":
        raise RuntimeError(
            f"USE_MATE_FA3 must be one of: auto, 0, 1; got {flag!r}"
        )
    # auto: only step in when the real flash-attn is missing and MATE is present.
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        pass
    else:
        return False
    try:
        import flash_attn_3  # noqa: F401
    except ImportError:
        return False
    return True


class _FlashAttnCompatDistribution(Distribution):
    """Minimal in-memory distribution advertising the FA2 version to TE."""

    def read_text(self, filename):
        if filename == "METADATA":
            return (
                "Metadata-Version: 2.1\n"
                "Name: flash-attn\n"
                f"Version: {_FA2_COMPAT_VERSION}\n"
                "Summary: MATE FlashAttention-3 exposed through the flash-attn 2 API\n"
            )
        return None

    def locate_file(self, path):
        return None


class _FlashAttnCompatFinder:
    """``importlib.metadata`` finder yielding the synthetic distribution.

    It also sits on ``sys.meta_path``, so it has to decline module imports
    instead of letting the import machinery trip over the missing hooks.
    """

    def find_distributions(self, context=DistributionFinder.Context()):
        if context.name in (None, "flash-attn", "flash_attn"):
            yield _FlashAttnCompatDistribution()

    def find_spec(self, fullname, path=None, target=None):
        return None

    def find_module(self, fullname, path=None):
        return None


def _register_distribution():
    for finder in sys.meta_path:
        if isinstance(finder, _FlashAttnCompatFinder):
            return
    sys.meta_path.append(_FlashAttnCompatFinder())


def _lse_to_te(softmax_lse):
    """MATE varlen returns [heads, total_t]; TE reads non-packed [b, heads, t]."""
    if softmax_lse is None:
        return None
    if softmax_lse.dtype != torch.float32:
        softmax_lse = softmax_lse.to(torch.float32)
    if softmax_lse.dim() == 2:
        softmax_lse = softmax_lse.unsqueeze(0)
    return softmax_lse.contiguous()


def _lse_to_mate(softmax_lse):
    """Inverse of :func:`_lse_to_te` for the varlen backward."""
    if softmax_lse.dim() == 3:
        if softmax_lse.size(0) != 1:
            raise NotImplementedError(
                "MATE varlen backward expects [heads, total_tokens]; got batched "
                f"softmax_lse with shape {tuple(softmax_lse.shape)}"
            )
        softmax_lse = softmax_lse.squeeze(0)
    if softmax_lse.dtype != torch.float32:
        softmax_lse = softmax_lse.to(torch.float32)
    return softmax_lse.contiguous()


_bind_tvm_stream = None
_tvm_stream_unavailable = False


def _sync_tvm_stream():
    """Point TVM's device stream at Torch's current MUSA stream.

    MATE's kernels are TileLang kernels launched through TVM. TileLang's tvm_ffi
    backend never forwards Torch's stream (``get_current_stream_functor`` is
    commented out upstream), so the kernels would run on TVM's own stream while
    TE drives context-parallel attention on ``cp_stream`` -- the resulting race
    corrupts dq/dk/dv and surfaces as an async "read write no barrier" fault a
    few iterations later.
    """
    global _bind_tvm_stream, _tvm_stream_unavailable
    if _tvm_stream_unavailable:
        return
    try:
        if _bind_tvm_stream is None:
            import tilelang  # noqa: F401  (puts the vendored tvm on sys.path)
            import tvm
            import torch_musa

            devices = {}
            raw_stream = torch_musa._MUSAC._musa_getCurrentRawStream

            def bind():
                index = torch.musa.current_device()
                device = devices.get(index)
                if device is None:
                    device = tvm.musa(index)
                    devices[index] = device
                device.set_raw_stream(raw_stream(index))

            _bind_tvm_stream = bind
        _bind_tvm_stream()
    except Exception as exc:  # pragma: no cover - depends on the TileLang build
        _tvm_stream_unavailable = True
        logger.warning(
            "[musa_patch] cannot bind TVM's stream to Torch's current stream; MATE "
            "kernels may race with TE context-parallel streams (%s)",
            exc,
        )


def _contig(*tensors):
    """flash-attn normalises strides inside the kernel wrappers; MATE's tilelang
    backward only validates shapes, so non-contiguous views (TE hands them over
    unchanged on the FlashAttention path) would be read with the wrong strides."""
    return [
        x if x is None or x.is_contiguous() else x.contiguous() for x in tensors
    ]


def _check_unsupported(dropout_p, alibi_slopes, block_table):
    if dropout_p:
        raise NotImplementedError(
            f"MATE FA3 has no attention dropout; got dropout_p={dropout_p}"
        )
    if alibi_slopes is not None:
        raise NotImplementedError("MATE FA3 does not support ALiBi slopes")
    if block_table is not None:
        raise NotImplementedError("MATE FA3 paged KV is not wired into this shim")


def _window(window_size):
    if window_size is None:
        return (-1, -1)
    return (window_size[0], window_size[1])


def _build_interface():
    from flash_attn_3.interface import (
        _flash_attn_forward as _mate_fwd,
        flash_attn_func as _mate_flash_attn_func,
        flash_attn_varlen_func as _mate_flash_attn_varlen_func,
    )

    def _mate_bwd(*args, **kwargs):
        from mate.flash_attention.tilelang.flash_attention_varlen_bwd import (
            flashattn_varlen_bwd_interface,
        )

        return flashattn_varlen_bwd_interface(*args, **kwargs)

    def _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
        block_table=None,
        leftpad_k=None,
        seqused_k=None,
        out=None,
        **_ignored,
    ):
        _check_unsupported(dropout_p, alibi_slopes, block_table)
        _sync_tvm_stream()
        window_left, window_right = _window(window_size)
        out, softmax_lse, *_rest = _mate_fwd(
            q,
            k,
            v,
            out_=out,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_left,
            window_size_right=window_right,
            softcap=softcap,
        )
        # MUSA FA2 returned (out, softmax_lse, S_dmask, rng_state);
        # flash_attn_cp_compat re-pads that into the 8-tuple TE indexes.
        return out, _lse_to_te(softmax_lse), None, None

    def _flash_attn_varlen_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        rng_state=None,
        **_ignored,
    ):
        _check_unsupported(dropout_p, alibi_slopes, None)
        _sync_tvm_stream()
        q, k, v, out, dout = _contig(q, k, v, out, dout)
        dq_, dk_, dv_ = _mate_bwd(
            q,
            k,
            v,
            out,
            dout,
            _lse_to_mate(softmax_lse),
            max_seqlen_q,
            max_seqlen_k,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            is_causal=causal,
            window_size=_window(window_size),
            softcap=softcap,
            smscale=softmax_scale,
            is_bhsd=False,
            deterministic=deterministic,
        )
        # TE preallocates dq/dk/dv and reads them after the call.
        dq.copy_(dq_)
        dk.copy_(dk_)
        dv.copy_(dv_)
        return dq, dk, dv

    def _flash_attn_forward(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
        block_table=None,
        out=None,
        **_ignored,
    ):
        _check_unsupported(dropout_p, alibi_slopes, block_table)
        _sync_tvm_stream()
        window_left, window_right = _window(window_size)
        out, softmax_lse, *_rest = _mate_fwd(
            q,
            k,
            v,
            out_=out,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_left,
            window_size_right=window_right,
            softcap=softcap,
        )
        if softmax_lse is not None and softmax_lse.dtype != torch.float32:
            softmax_lse = softmax_lse.to(torch.float32)
        return out, softmax_lse, None, None

    def _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        rng_state=None,
        **_ignored,
    ):
        _check_unsupported(dropout_p, alibi_slopes, None)
        _sync_tvm_stream()
        q, k, v, out, dout = _contig(q, k, v, out, dout)
        dq_, dk_, dv_ = _mate_bwd(
            q,
            k,
            v,
            out,
            dout,
            softmax_lse.contiguous(),
            None,
            None,
            is_causal=causal,
            window_size=_window(window_size),
            softcap=softcap,
            smscale=softmax_scale,
            is_bhsd=False,
            deterministic=deterministic,
        )
        dq.copy_(dq_)
        dk.copy_(dk_)
        dv.copy_(dv_)
        return dq, dk, dv

    def flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        **_ignored,
    ):
        _check_unsupported(dropout_p, alibi_slopes, None)
        _sync_tvm_stream()
        return _mate_flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=_window(window_size),
            softcap=softcap,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )

    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
        **_ignored,
    ):
        _check_unsupported(dropout_p, alibi_slopes, block_table)
        _sync_tvm_stream()
        return _mate_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=_window(window_size),
            softcap=softcap,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
        )

    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.__dict__.update(
        {
            "_flash_attn_forward": _flash_attn_forward,
            "_flash_attn_backward": _flash_attn_backward,
            "_flash_attn_varlen_forward": _flash_attn_varlen_forward,
            "_flash_attn_varlen_backward": _flash_attn_varlen_backward,
            "flash_attn_func": flash_attn_func,
            "flash_attn_varlen_func": flash_attn_varlen_func,
        }
    )
    return interface


def install():
    """Expose MATE FA3 as ``flash_attn``; returns True when the shim is active."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if not mate_fa3_requested():
        return False

    interface = _build_interface()
    package = types.ModuleType("flash_attn")
    package.__path__ = []
    package.__version__ = _FA2_COMPAT_VERSION
    package.flash_attn_interface = interface
    package.flash_attn_func = interface.flash_attn_func
    package.flash_attn_varlen_func = interface.flash_attn_varlen_func

    sys.modules["flash_attn"] = package
    sys.modules["flash_attn.flash_attn_interface"] = interface
    _register_distribution()

    import flash_attn_3

    _INSTALLED = True
    logger.warning(
        "[musa_patch] flash-attn %s API is backed by MATE FA3 %s",
        _FA2_COMPAT_VERSION,
        getattr(flash_attn_3, "__version__", "unknown"),
    )
    return True
