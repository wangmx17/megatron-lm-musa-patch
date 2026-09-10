"""Host-only specialization contract tests; not a substitute for TE/MUSA A/B."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[1] / "musa_patch/te_padded_metadata.py"
spec = importlib.util.spec_from_file_location("te_padded_metadata", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Metadata:
    def __init__(self, device):
        self.device = SimpleNamespace(type=device)
    def __getitem__(self, key):
        return self


calls = []
def equal(a, b):
    calls.append((a, b))
    return True


torch = SimpleNamespace(equal=equal)


def entry(cu_seqlens_q, cu_seqlens_kv, cu_seqlens_q_padded, cu_seqlens_kv_padded):
    q = (cu_seqlens_q_padded is not None
         and not torch.equal(cu_seqlens_q_padded[:-1], cu_seqlens_q[:-1]))
    kv = (cu_seqlens_kv_padded is not None
          and not torch.equal(cu_seqlens_kv_padded[:-1], cu_seqlens_kv[:-1]))
    return q, kv


def unsupported(value):
    return torch.equal(value, value)


class SpecializationTests(unittest.TestCase):
    def setUp(self):
        calls.clear()
        self.function = module.specialize(entry)

    def test_musa_uses_padded_path_without_equal(self):
        x = Metadata("musa")
        self.assertEqual(self.function(x, x, x, x), (True, True))
        self.assertEqual(calls, [])

    def test_no_padded_metadata_is_unchanged(self):
        x = Metadata("musa")
        self.assertEqual(self.function(x, x, None, None), (False, False))
        self.assertEqual(calls, [])

    def test_non_musa_preserves_equal(self):
        x = Metadata("cuda")
        self.assertEqual(self.function(x, x, x, x), (False, False))
        self.assertEqual(len(calls), 2)

    def test_unrecognized_source_fails_closed(self):
        with self.assertRaises(RuntimeError):
            module.specialize(unsupported)


if __name__ == "__main__":
    unittest.main()
