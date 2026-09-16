"""Isolated single-device compatibility tests; does not start training."""
import json
import torch
import musa_patch
from emerging_optimizers.orthogonalized_optimizers import Muon
from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy
from megatron.core.models.common.language_module.language_module import te_parallel_cross_entropy
assert te_parallel_cross_entropy is not None

torch.manual_seed(1234)
torch.set_float32_matmul_precision('medium')
x = torch.randn(128, 64, dtype=torch.float32)
cpu_ns = newton_schulz(x.clone(), steps=5, coefficient_type='quintic')
gpu_ns = newton_schulz(x.musa(), steps=5, coefficient_type='quintic')
p = torch.nn.Parameter(x.musa())
opt = Muon([p], lr=2e-8, momentum=.9, nesterov=False,
           coefficient_type='quintic', num_ns_steps=5,
           scale_mode='spectral', extra_scale_factor=1., use_syrk=False)
p.grad = torch.randn_like(p)
opt.step()
torch.musa.synchronize()
print('MUON', json.dumps({'finite': bool(torch.isfinite(p).all()),
    'ns_cpu_musa_max_abs': (gpu_ns.cpu()-cpu_ns).abs().max().item()}), flush=True)
for dtype in [torch.float32, torch.bfloat16]:
    try:
        a = torch.randn(8, 1, 1024, device='musa', dtype=dtype, requires_grad=True)
        b = a.detach().float().clone().requires_grad_()
        target = torch.randint(0, 1024, (8,1), device='musa')
        loss = te_parallel_cross_entropy(a, target, None)
        ref = torch.nn.functional.cross_entropy(b.reshape(-1,1024), target.flatten(), reduction='none').reshape(8,1)
        # Exercise stride-zero sum gradients; bridge must pack these for TE.
        loss.sum().backward()
        ref.sum().backward()
        torch.musa.synchronize()
        err = (a.grad.float()-b.grad).abs().max().item()
        assert err < (1e-6 if dtype == torch.float32 else 0.004), err
        print('TE_CE', json.dumps({'dtype':str(dtype),
            'loss_max_abs':(loss-ref).abs().max().item(),
            'grad_max_abs':(a.grad.float()-b.grad).abs().max().item(),
            'finite':bool(torch.isfinite(a.grad).all())}), flush=True)
    except Exception as e:
        print('TE_CE_FAILED', str(dtype), repr(e), flush=True)
        raise
print('peak_allocated_bytes', torch.musa.max_memory_allocated(), flush=True)
