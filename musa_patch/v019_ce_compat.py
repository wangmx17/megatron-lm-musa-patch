"""Isolated TE 2.0 bridge for the v0.19 FP32 CE workspace contract.

Preserves the installed fused TE kernels; no site-packages source is changed.
The saved softmax gradient is FP32, then autograd casts it to logits dtype.
This can cost extra memory relative to the v0.16 BF16 workspace.
"""
import torch
import transformer_engine.pytorch.cross_entropy as ce
import transformer_engine.pytorch.triton.cross_entropy as kernels


class FP32WorkspaceCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, label_smoothing=0., reduce_loss=False,
                dist_process_group=None):
        # TE forward overwrites its input with dLoss/dLogits. Own that storage.
        workspace = logits.to(dtype=torch.float32, copy=True)
        loss, derivative = kernels.cross_entropy_forward(
            workspace, labels, label_smoothing, reduce_loss, dist_process_group)
        if derivative.dtype != torch.float32:
            raise RuntimeError('TE CE compatibility requires FP32 saved derivative')
        ctx.save_for_backward(derivative)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        (derivative,) = ctx.saved_tensors
        # sum()/expand can provide stride-zero gradients; TE assumes dense input.
        result = kernels.cross_entropy_backward(derivative, grad_loss.contiguous())
        return result, None, None, None, None


ce.parallel_cross_entropy = FP32WorkspaceCrossEntropy.apply
ce.PARALLEL_CROSS_ENTROPY_FP32_GRAD = True
# Megatron may already have bound the original function during patch imports.
import megatron.core.extensions.transformer_engine as te_ext
te_ext.parallel_cross_entropy = ce.parallel_cross_entropy


def te_parallel_cross_entropy(logits, labels, tp_group, is_cg_capturable=False):
    if is_cg_capturable:
        raise NotImplementedError('TE 2.0 bridge is not CUDA-graph capturable')
    return ce.parallel_cross_entropy(logits, labels, 0., False, tp_group)


te_ext.te_parallel_cross_entropy = te_parallel_cross_entropy
from megatron.core.models.common.language_module import language_module
language_module.te_parallel_cross_entropy = te_parallel_cross_entropy
