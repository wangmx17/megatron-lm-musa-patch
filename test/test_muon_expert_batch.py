"""Compare the adapter against the actual compatible Megatron Muon source.

Set MEGATRON_MUON_SOURCE to its muon.py; optionally MUON_TEST_DEVICE=musa.
This checks complete tiny-optimizer updates, not model convergence.
"""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

source = os.environ.get('MEGATRON_MUON_SOURCE')
if not source:
    raise RuntimeError('Set MEGATRON_MUON_SOURCE to the compatible Megatron muon.py')
base = load('megatron.core.optimizer.muon', source)
candidate = load('muon_expert_batch', Path(__file__).resolve().parents[1] / 'musa_patch/muon_expert_batch.py')
device = os.environ.get('MUON_TEST_DEVICE', 'cpu')
if device == 'musa':
    import torch_musa
    torch.musa.set_device(0)
os.environ['MUON_BATCH_NS'] = '1'
os.environ['MUON_BATCH_NS_MAX_B'] = '8'
os.environ['MUON_DP1_LOW_MEMORY'] = '0'
os.environ['MUON_FUSED_POINTWISE'] = '0'


class MuonTests(unittest.TestCase):
    def test_three_complete_steps_mixed_shapes_and_adam(self):
        torch.manual_seed(71)
        shapes = [(8, 16)] * 9 + [(16, 8)] * 2 + [(4,)]
        values = [torch.randn(shape, device=device) for shape in shapes]
        a = [torch.nn.Parameter(x.clone()) for x in values]
        b = [torch.nn.Parameter(x.clone()) for x in values]
        def groups(params):
            return [dict(params=params[:-1], use_muon=True, is_expert_parallel=True),
                    dict(params=params[-1:], use_muon=False, is_expert_parallel=False)]
        oa = base.Muon(groups(a), lr=2e-4)
        ob = candidate.MuonExpertBatch(groups(b), lr=2e-4)
        for step in range(3):
            for x, y in zip(a, b):
                gradient = torch.randn_like(x)
                x.grad, y.grad = gradient.clone(), gradient.clone()
            oa.step()
            ob.step()
            errors = []
            for x, y in zip(a, b):
                errors.append((x-y).abs().max().item())
                torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5)
                self.assertEqual(oa.state[x].keys(), ob.state[y].keys())
                for key in oa.state[x]:
                    torch.testing.assert_close(oa.state[x][key], ob.state[y][key], rtol=0, atol=0)
            print('step', step+1, 'max_parameter_abs_error', max(errors), flush=True)
        self.assertTrue(ob.te_expert_batch_ns_observed)

    def test_distributed_eligibility_guards(self):
        p = torch.nn.Parameter(torch.zeros(8, 16, device=device))
        optimizer = candidate.MuonExpertBatch([dict(params=[p],use_muon=True,is_expert_parallel=True)])
        group = optimizer.param_groups[0]
        x = torch.zeros_like(p)
        self.assertTrue(optimizer._can_batch_te_expert_param(p, x, group))
        self.assertFalse(optimizer._can_batch_te_expert_param(p, x.unsqueeze(0), group))
        optimizer.distributed_mode = True
        optimizer.dist_world_size = 1
        meta = SimpleNamespace(tp_split_dim=-1, shape=(8,16), local_range=(0,128), global_range=(0,128))
        optimizer.dist_metas = {p: meta}
        self.assertTrue(optimizer._can_batch_te_expert_param(p, x, group))
        optimizer.dist_world_size = 2
        self.assertFalse(optimizer._can_batch_te_expert_param(p, x, group))
        optimizer.dist_world_size = 1
        meta.local_range = (0,64)
        self.assertFalse(optimizer._can_batch_te_expert_param(p, x, group))
        meta.local_range = (0,128)
        meta.tp_split_dim = 0
        self.assertFalse(optimizer._can_batch_te_expert_param(p, x, group))


if __name__ == '__main__':
    unittest.main()
