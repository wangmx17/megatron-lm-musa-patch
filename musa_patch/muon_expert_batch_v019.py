"""Opt-in v0.19 Muon batching of independent, rank-local expert matrices."""
import inspect
import os
from collections import OrderedDict
from functools import wraps

import torch


def batched_ns(x, steps, coefficient_type, precision):
    """Preserve EO normalization, coefficient order, and per-step BF16 rounding."""
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import (
        _COEFFICIENT_SETS, get_coefficient_iterator)
    if x.ndim != 3 or x.dtype != torch.float32:
        raise ValueError("Expected stacked FP32 matrices")
    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT
    x = torch.nn.functional.normalize(x, p=2, dim=(-2, -1), eps=1e-7)
    if precision == "medium":
        x = x.to(torch.bfloat16)
    mode = "repeat_last" if coefficient_type == "polar_express" else "cycle"
    for a, b, c in get_coefficient_iterator(
        steps, _COEFFICIENT_SETS[coefficient_type], mode=mode
    ):
        gram = torch.bmm(x, x.mT)
        poly = torch.baddbmm(gram, gram, gram, alpha=c, beta=b)
        x = torch.baddbmm(x, poly, x, alpha=1.0, beta=a)
    x = x.float()
    return x.mT if transpose else x


def install():
    from emerging_optimizers import utils
    from emerging_optimizers.orthogonalized_optimizers import \
        get_muon_scale_factor
    from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon

    cls = TensorParallelMuon
    if getattr(cls, "_musa_expert_batch_installed", False):
        return
    original_init, original_step = cls.__init__, cls.step
    signature = inspect.signature(original_init)

    @wraps(original_init)
    def init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        original_init(self, *args, **kwargs)
        self._musa_ns_config = {
            k: bound.arguments[k]
            for k in ("num_ns_steps", "coefficient_type", "scale_mode", "extra_scale_factor")
        }

    def eligible(self, group, p):
        return (
            group.get("is_expert_parallel", False)
            and getattr(p, "expert_tp", False)
            and self.tp_mode == "blockwise"
            and not getattr(p, "is_gtp_weight_remat", False)
            and not (self.split_qkv and self.is_qkv_fn(p))
            and p.ndim == 2
            and p.dtype == torch.float32
            and p.grad.layout == torch.strided
        )

    @torch.no_grad()
    def step(self, closure=None):
        # Do not intercept AdaptiveMuon or other subclasses' update semantics.
        if type(self) is not cls:
            return original_step(self, closure)
        chunk_size = int(os.getenv("MUON_TE_EXPERT_BATCH_SIZE", "8"))
        if chunk_size < 2:
            return original_step(self, closure)
        loss = None if closure is None else closure()
        cfg = self._musa_ns_config
        for group in self.param_groups:
            self._init_group(group)
            buckets = OrderedDict()
            for p in group["params"]:
                if p.grad is None:
                    continue
                key = (tuple(p.shape), p.dtype, p.device) if eligible(self, group, p) else ("single", id(p))
                buckets.setdefault(key, []).append(p)
            for params in buckets.values():
                for start in range(0, len(params), chunk_size):
                    chunk = params[start:start + chunk_size]
                    grads = []
                    for p in chunk:
                        grad = p.grad
                        self._apply_weight_decay_inplace(p, grad, group["lr"], group["weight_decay"])
                        momentum = self.state[p]["momentum_buffer"]
                        momentum.lerp_(grad, 1 - group["momentum"])
                        grads.append(grad.lerp(momentum, group["momentum"]) if self.nesterov else momentum)
                    with utils.fp32_matmul_precision(self.fp32_matmul_prec):
                        if len(chunk) > 1:
                            with torch.autograd.profiler.record_function("muon_te_expert_batched_ns"):
                                updates = batched_ns(
                                    torch.stack(grads), cfg["num_ns_steps"],
                                    cfg["coefficient_type"], self.fp32_matmul_prec,
                                )
                                scale = get_muon_scale_factor(
                                    chunk[0].shape[0], chunk[0].shape[1], mode=cfg["scale_mode"],
                                )
                                updates = updates * scale * cfg["extra_scale_factor"]
                                updates = updates.unbind(0)
                        else:
                            kwargs = {k: v for k, v in group.items() if k != "params"}
                            updates = [self.orthogonalize(chunk[0], grads[0], **kwargs)]
                    for p, update in zip(chunk, updates):
                        self.pre_weight_update_fn_inplace(p, update)
                        p.add_(update, alpha=-group["lr"])
                        self.post_weight_update_fn_inplace(p)
        return loss

    cls.__init__ = init
    cls.step = step
    cls._musa_expert_batch_installed = True
