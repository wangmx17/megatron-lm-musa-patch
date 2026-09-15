"""Numerical tests for MUSA device-side packed THD RoPE metadata."""

import importlib.util
import os
from pathlib import Path
import unittest

import torch


MODULE_PATH = Path(
    os.environ.get(
        "THD_ROPE_DEVICE_MODULE",
        Path(__file__).parents[1] / "musa_patch/thd_rope_device.py",
    )
)
SPEC = importlib.util.spec_from_file_location("thd_rope_device_under_test", MODULE_PATH)
DEVICE_ROPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEVICE_ROPE)


class FakeCPGroup:
    def __init__(self, size, rank):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def reference_thd_rope(t, lengths, freqs, interleaved, cp_size, cp_rank):
    outputs = []
    local_lengths = [length // cp_size for length in lengths]
    for tensor, full_length in zip(torch.split(t, local_lengths), lengths):
        if cp_size > 1:
            half_local_length = full_length // cp_size // 2
            local_freqs = torch.cat(
                (
                    freqs[cp_rank * half_local_length : (cp_rank + 1) * half_local_length],
                    freqs[
                        full_length - (cp_rank + 1) * half_local_length :
                        full_length - cp_rank * half_local_length
                    ],
                )
            )
        else:
            local_freqs = freqs[:full_length]
        outputs.append(
            torch.rope(
                tensor.unsqueeze(1),
                local_freqs.squeeze(1).squeeze(1),
                interleaved,
                False,
                False,
            ).squeeze(1)
        )
    return torch.cat(outputs)


class TestTHDRopeDevice(unittest.TestCase):
    def test_cpu_tensors_use_fallback(self):
        t = torch.empty((8, 1, 16))
        cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
        freqs = torch.empty((8, 1, 1, 16))
        self.assertFalse(DEVICE_ROPE.device_thd_rope_supported(t, cu_seqlens, freqs))

    @unittest.skipUnless(
        hasattr(torch, "musa") and torch.musa.is_available() and hasattr(torch, "rope"),
        "requires MUSA and native torch.rope",
    )
    def test_packed_variable_length_forward_and_backward(self):
        torch.manual_seed(1234)
        lengths = [16, 32, 48]
        cu_seqlens = torch.tensor([0, 16, 48, 96], device="musa", dtype=torch.int32)
        cases = 0

        for cp_size in (1, 2, 4):
            for cp_rank in range(cp_size):
                group = FakeCPGroup(cp_size, cp_rank)
                for heads in (1, 16):
                    for interleaved in (False, True):
                        t = torch.randn(
                            (96 // cp_size, heads, 128),
                            device="musa",
                            dtype=torch.bfloat16,
                            requires_grad=True,
                        )
                        reference_t = t.detach().clone().requires_grad_(True)
                        freqs = torch.randn(
                            (48, 1, 1, 128), device="musa", dtype=torch.float32
                        )
                        expected = reference_thd_rope(
                            reference_t, lengths, freqs, interleaved, cp_size, cp_rank
                        )
                        actual = DEVICE_ROPE.apply_rotary_pos_emb_thd_torch_rope_device(
                            t, cu_seqlens, freqs, interleaved, group
                        )
                        gradient = torch.randn_like(actual)
                        actual.backward(gradient)
                        expected.backward(gradient)

                        self.assertEqual((actual - expected).abs().max().item(), 0)
                        self.assertEqual((t.grad - reference_t.grad).abs().max().item(), 0)
                        self.assertFalse(
                            DEVICE_ROPE.device_thd_rope_supported(t, cu_seqlens, freqs[..., :64])
                        )
                        cases += 1

        self.assertEqual(cases, 28)


if __name__ == "__main__":
    unittest.main(verbosity=2)
