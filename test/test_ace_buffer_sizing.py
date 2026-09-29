import importlib.util
from pathlib import Path
import unittest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "musa_patch"
    / "deepep_ace"
    / "buffer_sizing.py"
)
_SPEC = importlib.util.spec_from_file_location("ace_buffer_sizing", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
compact_ace_nvl_bytes = _MODULE.compact_ace_nvl_bytes
select_ace_nvl_bytes = _MODULE.select_ace_nvl_bytes


class TestAceBufferSizing(unittest.TestCase):
    def test_default_floor_is_conservative(self):
        self.assertEqual(compact_ace_nvl_bytes(8, 160), 1 << 20)

    def test_unaudited_geometry_is_rejected(self):
        for args in ((0, 160), (8, 0), (8, 161), (16, 160)):
            with self.subTest(args=args), self.assertRaises(RuntimeError):
                compact_ace_nvl_bytes(*args)

    def test_disabled_path_preserves_legacy_hint(self):
        self.assertEqual(
            select_ace_nvl_bytes(352_784_896, 0, 8, 160, "any"),
            (352_784_896, "legacy-hint"),
        )

    def test_compact_path_requires_audited_build(self):
        self.assertEqual(
            select_ace_nvl_bytes(
                352_784_896, 1, 8, 160, "1.1.0+0bd46de"
            ),
            (1 << 20, "compact-metadata"),
        )
        with self.assertRaises(RuntimeError):
            select_ace_nvl_bytes(
                352_784_896, 1, 8, 160, "different-build"
            )


if __name__ == "__main__":
    unittest.main()
