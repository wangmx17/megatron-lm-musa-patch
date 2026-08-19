"""Compatibility aliases for Megatron-LM v0.16 process-group refactors.

Megatron-LM v0.16 removed the v0.14 process-group helper dataclasses
``GradCommProcessGroups``, ``GradFinalizeProcessGroups`` and
``ModelCommProcessGroups`` and unified them into a single
``ProcessGroupCollection`` (passed as the ``pg_collection`` argument).

The MUSA patch modules were written against v0.14 and only use these names as
*type annotations* (the runtime monkey-patches are gated behind env switches
such as ``USE_MUSA_ROUTER``/``ENABLE_PROFILER``). To let those modules import
cleanly under v0.16 without changing any runtime behaviour, we alias the old
names to ``ProcessGroupCollection``. When actually running on v0.14 the real
classes are used instead.
"""

try:  # Megatron-LM v0.16 unified process-group container
    from megatron.core.process_groups_config import ProcessGroupCollection

    GradCommProcessGroups = ProcessGroupCollection
    GradFinalizeProcessGroups = ProcessGroupCollection
    ModelCommProcessGroups = ProcessGroupCollection
except ImportError:  # pragma: no cover - fallback for genuine v0.14 trees
    from megatron.core.process_groups_config import (  # type: ignore
        GradCommProcessGroups,
        GradFinalizeProcessGroups,
        ModelCommProcessGroups,
    )

__all__ = [
    "GradCommProcessGroups",
    "GradFinalizeProcessGroups",
    "ModelCommProcessGroups",
]
