"""CPU-only tests for the always-on MATE GroupedLinear adapter helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parents[1] / "musa_patch" / "mate_grouped_gemm.py"
SPEC = importlib.util.spec_from_file_location("mate_grouped_gemm_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _mark_group(parameters):
    marker = id(parameters)
    for index, parameter in enumerate(parameters):
        parameter._mate_grouped_gemm_group = marker
        parameter._mate_grouped_gemm_index = index
        parameter._mate_grouped_gemm_size = len(parameters)


class MateGroupedGemmHelperTests(unittest.TestCase):
    def test_dependency_contract_is_pinned(self):
        self.assertEqual(MODULE.MATE_REQUIRED_VERSION, "0.2.7")
        self.assertEqual(MODULE.MATE_GEMM_BACKEND, "mubin")

    def test_is_packed_checks_storage_extent_and_adjacency(self):
        packed = torch.empty((3, 4, 8), dtype=torch.bfloat16)
        self.assertTrue(MODULE._is_packed(list(packed.unbind(0))))

        unpacked = [torch.empty((4, 8), dtype=torch.bfloat16) for _ in range(3)]
        self.assertFalse(MODULE._is_packed(unpacked))

    def test_reorder_marked_param_group_keeps_indices_paired(self):
        experts = [torch.nn.Parameter(torch.empty(2, 2)) for _ in range(3)]
        unmarked = torch.nn.Parameter(torch.empty(2, 2))
        _mark_group(experts)

        reordered, indices, changed = MODULE._reorder_marked_param_groups(
            [unmarked, *experts], [10, 20, 21, 22]
        )

        self.assertEqual(reordered, [unmarked, experts[2], experts[1], experts[0]])
        self.assertEqual(indices, [10, 22, 21, 20])
        self.assertEqual(changed, 1)

    def test_reorder_rejects_incomplete_expert_run(self):
        experts = [torch.nn.Parameter(torch.empty(2, 2)) for _ in range(3)]
        _mark_group(experts)

        with self.assertRaisesRegex(RuntimeError, "complete contiguous parameter run"):
            MODULE._reorder_marked_param_groups(experts[:2], [0, 1])

    def test_reorder_rejects_malformed_group_metadata(self):
        expert = torch.nn.Parameter(torch.empty(2, 2))
        expert._mate_grouped_gemm_group = id(expert)
        expert._mate_grouped_gemm_index = 0
        expert._mate_grouped_gemm_size = None

        with self.assertRaisesRegex(RuntimeError, "complete contiguous parameter run"):
            MODULE._reorder_marked_param_groups([expert], [0])

    def test_dynamic_contract_rejects_split_type_before_using_it(self):
        module = type(
            "Module",
            (),
            {
                "fp8": False,
                "fp8_calibration": False,
                "use_bias": False,
                "return_bias": False,
                "gemm_bias_unfused_add": False,
                "fuse_wgrad_accumulation": True,
                "num_gemms": 2,
            },
        )()
        inp = SimpleNamespace(
            device=SimpleNamespace(type="musa"),
            dtype=torch.bfloat16,
            is_contiguous=lambda: True,
        )

        reason = MODULE._dynamic_unsupported_reason(module, inp, None, False)

        self.assertEqual(reason, "split_type")

    def test_weight_contract_cache_is_invalidated_before_repacking(self):
        packed = torch.empty((2, 4, 8), dtype=torch.bfloat16)
        module = type("Module", (), {"num_gemms": 2})()
        module.weight0 = torch.nn.Parameter(packed[0])
        module.weight1 = torch.nn.Parameter(packed[1])
        module._mate_weight_contract_validated = True

        MODULE._pack_weights_before_ddp(module)

        self.assertFalse(module._mate_weight_contract_validated)

    def test_weight_contract_accepts_packed_bf16_fp32_main_grads(self):
        packed = torch.empty((2, 4, 8), dtype=torch.bfloat16)
        weights = [torch.nn.Parameter(weight) for weight in packed.unbind(0)]
        for weight in weights:
            weight.main_grad = torch.empty_like(weight, dtype=torch.float32)

        self.assertIsNone(MODULE._weight_contract_unsupported_reason(weights))

    def test_weight_contract_is_checked_once_and_rechecked_after_repack(self):
        packed = torch.empty((2, 4, 8), dtype=torch.bfloat16)
        module = type("Module", (), {"num_gemms": 2})()
        module.weight0 = torch.nn.Parameter(packed[0])
        module.weight1 = torch.nn.Parameter(packed[1])
        weights = [module.weight0, module.weight1]

        with mock.patch.object(
            MODULE, "_weight_contract_unsupported_reason", return_value=None
        ) as validate:
            MODULE._ensure_weight_contract(module, weights)
            MODULE._ensure_weight_contract(module, weights)
            self.assertEqual(validate.call_count, 1)

            MODULE._pack_weights_before_ddp(module)
            MODULE._ensure_weight_contract(module, weights)
            self.assertEqual(validate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
