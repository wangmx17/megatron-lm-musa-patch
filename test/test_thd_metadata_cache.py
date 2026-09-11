"""CPU tests: run directly without importing the MUSA patch package."""
import gc
import importlib.util
from pathlib import Path
import unittest
import weakref

import torch

path = Path(__file__).resolve().parents[1] / "musa_patch/thd_metadata_cache.py"
spec = importlib.util.spec_from_file_location("thd_metadata_cache", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CacheTests(unittest.TestCase):
    def test_hit_and_inplace_invalidation(self):
        cache = module.THDSequenceLengthCache()
        cu = torch.tensor([0, 4, 12], dtype=torch.int32)
        first = cache.get(cu)
        self.assertEqual(first, [4, 8])
        self.assertIs(cache.get(cu), first)
        cu[1] = 6
        self.assertEqual(cache.get(cu), [6, 6])
        self.assertIsNot(cache.get(cu), first)

    def test_alias_mutation_invalidates(self):
        cache = module.THDSequenceLengthCache()
        cu = torch.tensor([0, 4, 12], dtype=torch.int32)
        cache.get(cu)
        cu.view(-1)[1] = 8
        self.assertEqual(cache.get(cu), [8, 4])

    def test_identity_lifetime_and_bound(self):
        cache = module.THDSequenceLengthCache(max_entries=2)
        values = [torch.tensor([0, i]) for i in (2, 3, 4)]
        for value in values:
            cache.get(value)
        self.assertEqual(len(cache._entries), 2)
        self.assertNotIn(id(values[0]), cache._entries)
        ref = weakref.ref(values[1])
        del values[1]
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(len(cache._entries), 1)

    def test_inference_tensors_are_not_cached(self):
        cache = module.THDSequenceLengthCache()
        with torch.inference_mode():
            cu = torch.tensor([0, 4, 12])
            self.assertEqual(cache.get(cu), [4, 8])
            cu[1] = 6
            self.assertEqual(cache.get(cu), [6, 6])
        self.assertFalse(cache._entries)


if __name__ == "__main__":
    unittest.main()
