"""Sizing helpers for the legacy metadata area used by DeepEP ACE."""

from __future__ import annotations


_LEGACY_BUFFER_ALIGNMENT_BYTES = 256
_INT32_BYTES = 4
COMPACT_NVL_FLOOR_BYTES = 1 << 20
COMPACT_NVL_SUPPORTED_VERSION = "1.1.0+0bd46de"
COMPACT_NVL_AUDITED_NUM_RANKS = 8
COMPACT_NVL_AUDITED_NUM_EXPERTS = 160


def compact_ace_nvl_bytes(
    num_ranks: int,
    num_experts: int,
) -> int:
    """Return a conservative aligned size for ACE legacy metadata.

    ACE payloads use separately registered windows. The retained legacy area
    carries a rank matrix and local-expert counts for ``ace_notify_dispatch``.
    A 1 MiB default floor leaves ample room above the exact int32 payload.
    """
    expected = (COMPACT_NVL_AUDITED_NUM_RANKS, COMPACT_NVL_AUDITED_NUM_EXPERTS)
    actual = (num_ranks, num_experts)
    if actual != expected:
        raise RuntimeError(
            "compact ACE NVL sizing is bound to the audited MiniCPM5 topology: "
            f"expected_ranks_experts={expected}, actual_ranks_experts={actual}"
        )
    local_experts = num_experts // num_ranks
    required_bytes = num_ranks * (num_ranks + local_experts) * _INT32_BYTES
    requested_bytes = max(required_bytes, COMPACT_NVL_FLOOR_BYTES)
    alignment = _LEGACY_BUFFER_ALIGNMENT_BYTES
    return ((requested_bytes + alignment - 1) // alignment) * alignment


def select_ace_nvl_bytes(
    legacy_hint_bytes: int,
    compact_enabled: int,
    num_ranks: int,
    num_experts: int,
    installed_version: str,
) -> tuple[int, str]:
    """Select the default legacy hint or the audited compact size."""
    if legacy_hint_bytes < 0:
        raise ValueError(f"legacy_hint_bytes must be non-negative, got {legacy_hint_bytes}")
    if compact_enabled not in (0, 1):
        raise ValueError(
            "DEEPEP_ACE_COMPACT_NVL_BUFFER must be 0 or 1, "
            f"got {compact_enabled}"
        )
    if not compact_enabled:
        return legacy_hint_bytes, "legacy-hint"
    if installed_version != COMPACT_NVL_SUPPORTED_VERSION:
        raise RuntimeError(
            "compact ACE NVL sizing is bound to the audited DeepEP build: "
            f"expected={COMPACT_NVL_SUPPORTED_VERSION}, installed={installed_version}"
        )
    return (
        compact_ace_nvl_bytes(num_ranks, num_experts),
        "compact-metadata",
    )
