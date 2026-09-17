"""Standalone MUSA optimizer update probe; does not launch training."""
# isort: skip_file
import os
os.environ["MUON_TE_EXPERT_BATCH_NS"] = "0"
import musa_patch  # bootstrap before EO/Megatron imports
import torch
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon

from musa_patch.muon_expert_batch_v019 import install

original_step = TensorParallelMuon.step
install()
torch.manual_seed(1234)
for nesterov in (False, True):
    params = [torch.nn.Parameter(torch.randn(512, 2048, device="musa") * 0.02)
              for _ in range(4)]
    copies = [torch.nn.Parameter(p.detach().clone()) for p in params]
    for p in params + copies:
        p.expert_tp = True
    kwargs = dict(lr=2e-5, momentum=0.9, nesterov=nesterov,
                  weight_decay=0.1, tp_mode="blockwise",
                  coefficient_type="quintic", num_ns_steps=5,
                  scale_mode="spectral", extra_scale_factor=1.0,
                  fp32_matmul_prec="medium")
    ref = TensorParallelMuon([dict(params=params, is_expert_parallel=True)], **kwargs)
    opt = TensorParallelMuon([dict(params=copies, is_expert_parallel=True)], **kwargs)
    for step in range(10):
        for i, (p, q) in enumerate(zip(params, copies)):
            if step == 3 and i == 1:
                p.grad = q.grad = None
            else:
                grad = torch.randn_like(p) if step else torch.zeros_like(p)
                p.grad, q.grad = grad.clone(), grad.clone()
        original_step(ref)
        opt.step()
        errors = [(p-q).abs().max().item() for p, q in zip(params, copies)]
        momentum_errors = [
            (ref.state[p]["momentum_buffer"]-opt.state[q]["momentum_buffer"]).abs().max().item()
            for p, q in zip(params, copies)
        ]
        assert max(momentum_errors) == 0, momentum_errors
        assert max(errors) < 1e-5, errors
        assert all(torch.isfinite(q).all().item() for q in copies)
        print("nesterov", nesterov, "step", step+1,
              "max_weight_abs", max(errors), "max_momentum_abs", max(momentum_errors), flush=True)
print("PROBE_COMPLETED; training precision acceptance remains pending", flush=True)
