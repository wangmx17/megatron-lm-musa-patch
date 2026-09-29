"""Static contract for the MiniCPM5 validated default performance stack."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "examples/minicpm5/run_16a3b.sh"


class MiniCPM5DefaultBestStack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = LAUNCHER.read_text(encoding="utf-8")

    def assert_default(self, name, value):
        pattern = rf"^export {re.escape(name)}=\$\{{{re.escape(name)}:-{re.escape(value)}\}}$"
        self.assertRegex(self.source, re.compile(pattern, re.MULTILINE))

    def assert_forwarded(self, name, value):
        expected = f"{name}='${{{name}:-{value}}}' \\\n"
        self.assertIn(expected, self.source)

    def test_retained_stack_defaults(self):
        expected = {
            "ENABLE_CE_TE": "1",
            "ENABLE_MANUAL_GC": "1",
            "ENABLE_ROPE_FUSION": "1",
            "ENABLE_MOE_ROUTER_FUSION": "1",
            "ENABLE_GRAD_ACCUM_FUSION": "1",
            "ENABLE_DEEPEP_ENV": "1",
            "USE_GROUPED_GEMM": "1",
            "DISABLE_MOE_FUSIONS": "0",
            "DISABLE_PERMUTE_FUSION": "0",
            "MOE_DEEPEP_NUM_SMS": "56",
            "MUON_BATCH_NS": "1",
            "MUON_BATCH_NS_MAX_B": "8",
            "MUON_TE_EXPERT_BATCH_NS": "1",
            "MUSA_FUSED_ROUTE_CONVERSION": "1",
            "NVTE_BATCH_MHA_P2P_COMM": "1",
            "MUSA_CP_FORWARD_BATCH_OVERLAP": "1",
            "MUSA_CP_BACKWARD_BATCH_OVERLAP": "1",
            "DISABLE_RECOMPUTE": "1",
            "MCCL_MIN_NCHANNELS": "16",
            "MCCL_MAX_NCHANNELS": "16",
            "MCCL_BUFFSIZE": "16777216",
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assert_default(name, value)

    def test_new_rank_local_flags_are_forwarded(self):
        expected = {
            "USE_GROUPED_GEMM": "1",
            "DISABLE_MOE_FUSIONS": "0",
            "DISABLE_PERMUTE_FUSION": "0",
            "MUON_BATCH_NS_MAX_B": "8",
            "MUON_TE_EXPERT_BATCH_NS": "1",
            "MUSA_FUSED_ROUTE_CONVERSION": "1",
            "NVTE_BATCH_MHA_P2P_COMM": "1",
            "MUSA_CP_FORWARD_BATCH_OVERLAP": "1",
            "MUSA_CP_BACKWARD_BATCH_OVERLAP": "1",
            "MUSA_TE_THD_LSE_FP32": "disable",
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assert_forwarded(name, value)

    def test_unretained_candidates_stay_off(self):
        for name in (
            "ENABLE_DEEPEP",
            "ENABLE_SHARED_EXPERT_OVERLAP",
            "ENABLE_RMSNORM_FUSION",
            "MUSA_FA_AUX_FUSION",
            "NVTE_MUSA_THD_CP_CORRECTION_FUSION",
        ):
            with self.subTest(name=name):
                self.assert_default(name, "0")


if __name__ == "__main__":
    unittest.main()
