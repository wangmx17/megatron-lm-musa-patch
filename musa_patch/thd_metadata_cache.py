"""Bounded CPU metadata cache for packed THD RoPE.

Never identify a tensor by its storage address alone. Tensor identity prevents
allocator reuse collisions; the version counter invalidates in-place updates.
Inference tensors without version tracking deliberately use the uncached path.
"""

from collections import OrderedDict
import weakref


class THDSequenceLengthCache:
    def __init__(self, max_entries=64):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._entries = OrderedDict()

    @staticmethod
    def _read(cu_seqlens):
        return (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

    def get(self, cu_seqlens):
        try:
            version = cu_seqlens._version
        except RuntimeError:
            return self._read(cu_seqlens)
        key = id(cu_seqlens)
        entry = self._entries.get(key)
        if entry is not None and entry[0]() is cu_seqlens and entry[1] == version:
            self._entries.move_to_end(key)
            return entry[2]

        lengths = self._read(cu_seqlens)

        def forget(ref):
            current = self._entries.get(key)
            if current is not None and current[0] is ref:
                self._entries.pop(key, None)

        self._entries[key] = (weakref.ref(cu_seqlens, forget), version, lengths)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return lengths


_cache = THDSequenceLengthCache()


def thd_seqlens_cpu(cu_seqlens):
    """Return per-sequence CPU lengths; callers must not modify the returned list."""
    return _cache.get(cu_seqlens)
