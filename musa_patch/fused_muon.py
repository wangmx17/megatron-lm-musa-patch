"""Optional MUSA Triton kernels for Muon's bandwidth-bound pointwise stages."""

import logging

import torch

from megatron.core.optimizer.muon import register_muon_fused_pointwise_backend


logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _muon_prepare_kernel(
        grad_ptr,
        momentum_buffer_ptr,
        ns_input_ptr,
        numel,
        momentum,
        NESTEROV: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
        old_momentum = tl.load(momentum_buffer_ptr + offsets, mask=mask).to(tl.float32)
        new_momentum = old_momentum * momentum + grad
        tl.store(momentum_buffer_ptr + offsets, new_momentum, mask=mask)
        if NESTEROV:
            ns_input = grad + momentum * new_momentum
        else:
            ns_input = new_momentum
        # ns_input_ptr is BF16, so the store performs the only required cast and
        # avoids materializing the eager FP32 Nesterov temporary.
        tl.store(ns_input_ptr + offsets, ns_input, mask=mask)


    @triton.jit
    def _muon_apply_kernel(
        param_ptr,
        update_ptr,
        numel,
        decay_scale,
        update_scale,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        param = tl.load(param_ptr + offsets, mask=mask).to(tl.float32)
        update = tl.load(update_ptr + offsets, mask=mask).to(tl.float32)
        param = param * decay_scale + update * update_scale
        tl.store(param_ptr + offsets, param, mask=mask)


def _is_musa_contiguous(tensor):
    return tensor.device.type == "musa" and tensor.is_contiguous()


def fused_prepare_muon_input(grad, momentum_buffer, momentum, nesterov):
    """Return BF16 NS input, or None when this kernel cannot handle the tensors."""
    if (
        not HAVE_TRITON
        or grad.dtype != torch.float32
        or momentum_buffer.dtype != torch.float32
        or grad.shape != momentum_buffer.shape
        or grad.device != momentum_buffer.device
        or not _is_musa_contiguous(grad)
        or not _is_musa_contiguous(momentum_buffer)
    ):
        return None

    ns_input = torch.empty_like(grad, dtype=torch.bfloat16)
    numel = grad.numel()
    if numel:
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        _muon_prepare_kernel[grid](
            grad,
            momentum_buffer,
            ns_input,
            numel,
            momentum,
            NESTEROV=nesterov,
            BLOCK_SIZE=512,
        )
    return ns_input


def fused_apply_muon_update(param, update, decay_scale, update_scale):
    """Apply decay and orthogonalized update together; return whether it ran."""
    if (
        not HAVE_TRITON
        or param.dtype != torch.float32
        or update.dtype not in (torch.bfloat16, torch.float32)
        or param.numel() != update.numel()
        or param.device != update.device
        or not _is_musa_contiguous(param)
        or not _is_musa_contiguous(update)
    ):
        return False

    numel = param.numel()
    if numel:
        grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
        _muon_apply_kernel[grid](
            param,
            update,
            numel,
            decay_scale,
            update_scale,
            BLOCK_SIZE=512,
        )
    return True


if HAVE_TRITON:
    # The apply kernel remains available for explicit microbenchmarks, but the
    # target S5000 operator gate showed a regression at the real DPxCP shard
    # size. Register only the prepare kernel until a backend upgrade is retested.
    register_muon_fused_pointwise_backend(fused_prepare_muon_input, None)
    logger.warning("Registered MUSA Triton Muon fused prepare backend; apply remains eager")
else:
    logger.warning("Triton is unavailable; Muon fused pointwise backend was not registered")
