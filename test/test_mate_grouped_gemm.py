"""CPU-only tests for the always-on MATE GroupedLinear adapter helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
