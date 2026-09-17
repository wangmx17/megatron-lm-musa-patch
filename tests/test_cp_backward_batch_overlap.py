"""CPU-only installer safety checks; training/gradient evidence is separate."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "musa_patch/cp_backward_batch_overlap.py"
spec = importlib.util.spec_from_file_location("cp_overlap_under_test", MODULE_PATH)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
FLAG = "MUSA_CP_BACKWARD_BATCH_OVERLAP"


class InstallerSafety(unittest.TestCase):
    def fake_modules(self):
        torch = types.ModuleType("torch")
        te = types.ModuleType("transformer_engine")
        pytorch = types.ModuleType("transformer_engine.pytorch")
        attention = types.ModuleType("transformer_engine.pytorch.attention")
        attention.__file__ = __file__
        te.pytorch = pytorch
        pytorch.attention = attention
        return {"torch": torch, "transformer_engine": te,
                "transformer_engine.pytorch": pytorch,
                "transformer_engine.pytorch.attention": attention}

    def test_default_off_does_not_import_torch_or_te(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(
                sys.modules, {"torch": None, "transformer_engine": None}):
            adapter.install()

    def test_separate_p2p_rejected(self):
        with patch.dict(os.environ, {FLAG: "1", "NVTE_BATCH_MHA_P2P_COMM": "0"}), \
                patch.dict(sys.modules, self.fake_modules()):
            with self.assertRaisesRegex(RuntimeError, "requires NVTE_BATCH"):
                adapter.install()

    def test_simultaneous_forward_backward_reaches_standard_checks(self):
        with patch.dict(os.environ, {
                FLAG: "1",
                "MUSA_CP_FORWARD_BATCH_OVERLAP": "1",
                "NVTE_BATCH_MHA_P2P_COMM": "0",
        }, clear=True), patch.dict(sys.modules, self.fake_modules()):
            with self.assertRaisesRegex(RuntimeError, "requires NVTE_BATCH"):
                adapter.install()

    def test_unknown_te_source_rejected(self):
        with patch.dict(os.environ, {FLAG: "1", "NVTE_BATCH_MHA_P2P_COMM": "1"}), \
                patch.dict(sys.modules, self.fake_modules()):
            with self.assertRaisesRegex(RuntimeError, "Unsupported TE"):
                adapter.install()

    def test_unrecognized_function_layout_rejected(self):
        with self.assertRaises(RuntimeError):
            adapter.build_source("def backward(ctx):\n    pass\n", "backward")

    def test_live_function_change_rejected_before_install(self):
        def original(ctx):
            return None
        modules = self.fake_modules()
        attention = modules["transformer_engine.pytorch.attention"]
        attention.AttnFuncWithCPAndKVP2P = types.SimpleNamespace(backward=original)
        with patch.dict(sys.modules, modules), \
                patch.object(adapter.inspect, "getsource", return_value="unused"), \
                patch.object(adapter, "build_source", return_value="def backward(ctx):\n    return None\n"):
            with self.assertRaisesRegex(RuntimeError, "Unexpected live TE"):
                adapter._install_experiment()
        self.assertIs(attention.AttnFuncWithCPAndKVP2P.backward, original)


if __name__ == "__main__":
    unittest.main()
