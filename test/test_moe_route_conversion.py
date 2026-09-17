"""MUSA forward/backward comparison for the submitted conversion kernel."""
import importlib.util
from pathlib import Path
import unittest

import torch
import torch_musa
import musa_patch  # Match training bootstrap, including TE's runtime library loading.

path = Path(__file__).resolve().parents[1] / 'musa_patch/moe_route_conversion.py'
spec = importlib.util.spec_from_file_location('moe_route_conversion', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipUnless(torch.musa.is_available(), 'requires MUSA')
class RouteTests(unittest.TestCase):
    def test_forward_and_probability_gradient(self):
        torch.musa.set_device(0)
        generator = torch.Generator().manual_seed(71)
        for rows, experts, topk in ((1, 5, 3), (37, 20, 16), (64, 32, 8)):
            indices_cpu = torch.stack([torch.randperm(experts, generator=generator)[:topk]
                                       for _ in range(rows)]).to(torch.int64)
            indices_cpu[0, :] = -1
            if rows > 1:
                indices_cpu[1, -2:] = -1
            indices = indices_cpu.to('musa')
            values = torch.randn(rows, topk, generator=generator)
            probs = values.to('musa').requires_grad_()
            routing, actual = module.fused_indices_to_multihot(indices, probs, experts)
            expected = torch.zeros(rows, experts)
            expected_routing = torch.zeros(rows, experts, dtype=torch.bool)
            for row in range(rows):
                for position in range(topk):
                    expert = int(indices_cpu[row, position])
                    if expert >= 0:
                        expected[row, expert] = values[row, position]
                        expected_routing[row, expert] = True
            torch.testing.assert_close(actual.detach().cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(routing.cpu(), expected_routing, rtol=0, atol=0)
            # Transposed, non-contiguous upstream gradient exercises the kernel's
            # explicit contiguous conversion; masked input gradients must be zero.
            weight = torch.randn(experts, rows, generator=generator).t()
            actual.backward(weight.to('musa'))
            expected_grad = torch.zeros_like(values)
            for row in range(rows):
                for position in range(topk):
                    expert = int(indices_cpu[row, position])
                    if expert >= 0:
                        expected_grad[row, position] = weight[row, expert]
            torch.testing.assert_close(probs.grad.cpu(), expected_grad, rtol=0, atol=0)
            print('shape', (rows, experts, topk), 'forward_and_gradient_exact', flush=True)


if __name__ == '__main__':
    unittest.main()
