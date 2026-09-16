"""Load the canonical Megatron v0.19 GPT pretraining helpers.

Some model examples are themselves named ``pretrain_gpt.py``. Importing the
canonical root entrypoint by module name from those scripts resolves back to
the example and creates a circular import. Resolve the canonical file from the
installed/imported ``megatron`` package instead, independent of ``sys.path``
ordering.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import megatron


_MODULE_NAME = "_mcore_v019_canonical_pretrain_gpt"


def _load_canonical_pretrain_gpt():
    module = sys.modules.get(_MODULE_NAME)
    if module is not None:
        return module

    package_file = getattr(megatron, "__file__", None)
    if package_file is not None:
        megatron_package_dir = Path(package_file).resolve().parent
    else:
        package_paths = list(getattr(megatron, "__path__", ()))
        if not package_paths:
            raise ImportError("Unable to locate the imported megatron package")
        megatron_package_dir = Path(package_paths[0]).resolve()

    module_path = megatron_package_dir.parent / "pretrain_gpt.py"
    if not module_path.is_file():
        raise ImportError(f"Canonical Megatron pretrain_gpt.py not found at {module_path}")

    spec = spec_from_file_location(_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load canonical Megatron entrypoint from {module_path}")

    module = module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


_CANONICAL_PRETRAIN_GPT = _load_canonical_pretrain_gpt()
_original_get_batch = _CANONICAL_PRETRAIN_GPT.get_batch


def _get_batch_with_baseline_thd(data_iterator, vp_stage=None):
    """Keep v0.16 local indexed-data one-span semantics, not SDK packing.

    Native CP has already selected the two balanced pieces of this single
    full sequence. Attach global boundaries, never rank-local RoPE positions.
    Reject unsupported layouts rather than silently label multi-doc data dense.
    """
    from megatron.training import get_args
    import torch
    args = get_args()
    batch = list(_original_get_batch(data_iterator, vp_stage))
    if getattr(args, 'use_span_based_attn', False) and batch[1] is None:
        if args.micro_batch_size != 1 or args.reset_position_ids or args.reset_attention_mask:
            raise ValueError('Baseline THD bridge requires MBS1 and no position/mask resets')
        if args.pipeline_model_parallel_size != 1:
            raise ValueError('Baseline THD bridge currently validated for PP1 only')
        device = batch[-1].device
        batch[1] = torch.tensor([0, args.seq_length], dtype=torch.int32, device=device)
        batch[7] = torch.tensor(args.seq_length, dtype=torch.int32, device=device)
    return tuple(batch)


_CANONICAL_PRETRAIN_GPT.get_batch = _get_batch_with_baseline_thd
forward_step = _CANONICAL_PRETRAIN_GPT.forward_step
train_valid_test_datasets_provider = _CANONICAL_PRETRAIN_GPT.train_valid_test_datasets_provider


__all__ = ["forward_step", "train_valid_test_datasets_provider"]
