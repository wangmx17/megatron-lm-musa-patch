"""Always-on MATE BF16 fast path for Transformer Engine GroupedLinear.

MATE owns routed-expert fprop and dgrad. Transformer Engine keeps wgrad and
continues to accumulate it into FP32 ``main_grad``. The adapter deliberately
does not include CPU affinity, deferred route counts, or MATE ragged-K wgrad.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Sequence

import torch


MATE_REQUIRED_VERSION = "0.2.7"
MATE_GEMM_BACKEND = "mubin"


def _rank() -> str:
    return os.getenv("RANK", "?")


def _log(message: str) -> None:
    print(f"[MATE_GROUPED_GEMM][rank{_rank()}] {message}", flush=True)


@functools.lru_cache(maxsize=1)
def load_mate_gemm():
    """Load the pinned public MATE GEMM API after torch_musa registers DLPack."""
    import torch_musa  # noqa: F401
    import mate
    from mate import gemm as mate_gemm

    source = Path(mate.__file__).resolve()
    version_file = source.parent.parent / "version.txt"
    source_version = version_file.read_text().strip() if version_file.is_file() else "unknown"
    if source_version != MATE_REQUIRED_VERSION:
        raise RuntimeError(
            f"MATE source version {source_version!r} does not match required "
            f"{MATE_REQUIRED_VERSION!r}; source={source}"
        )
    _log(f"loaded source={source} source_version={source_version}")
    return mate_gemm


def _is_packed(tensors: Sequence[torch.Tensor]) -> bool:
    if not tensors or not all(tensor.is_contiguous() for tensor in tensors):
        return False
    first = tensors[0]
    span = first.numel() * first.element_size()
    addresses_are_packed = all(
        tensor.shape == first.shape
        and tensor.dtype == first.dtype
        and tensor.device == first.device
        and tensor.data_ptr() == first.data_ptr() + index * span
        for index, tensor in enumerate(tensors)
    )
    required_bytes = len(tensors) * span
    available_bytes = (
        first.untyped_storage().nbytes() - first.storage_offset() * first.element_size()
    )
    return addresses_are_packed and available_bytes >= required_bytes


def _weights(module):
    return [getattr(module, f"weight{index}") for index in range(module.num_gemms)]


def _pack_weights_before_ddp(module) -> None:
    weights = _weights(module)
    for index, weight in enumerate(weights):
        weight._mate_grouped_gemm_group = id(module)
        weight._mate_grouped_gemm_index = index
        weight._mate_grouped_gemm_size = len(weights)
    if not weights or _is_packed(weights):
        return
    if any(
        weight.device.type != "musa" or weight.dtype != torch.bfloat16
        for weight in weights
    ):
        return
    packed = torch.empty(
        (len(weights), *weights[0].shape),
        dtype=weights[0].dtype,
        device=weights[0].device,
    )
    with torch.no_grad():
        for index, weight in enumerate(weights):
            packed[index].copy_(weight)
            weight.data = packed[index]


def _reorder_marked_param_groups(params, param_indices):
    """Counter Megatron's reverse DDP-buffer walk for packed expert weights."""
    params = list(params)
    param_indices = list(param_indices)
    out_params = []
    out_indices = []
    changed_groups = 0
    position = 0
    while position < len(params):
        param = params[position]
        group = getattr(param, "_mate_grouped_gemm_group", None)
        if group is None:
            out_params.append(param)
            out_indices.append(param_indices[position])
            position += 1
            continue
        end = position
        while (
            end < len(params)
            and getattr(params[end], "_mate_grouped_gemm_group", None) == group
        ):
            end += 1
        run_params = params[position:end]
        run_indices = param_indices[position:end]
        expected_size = getattr(run_params[0], "_mate_grouped_gemm_size", None)
        expert_indices = [
            getattr(item, "_mate_grouped_gemm_index", None) for item in run_params
        ]
        if (
            not isinstance(expected_size, int)
            or expected_size < 0
            or expected_size != len(run_params)
            or expert_indices != list(range(expected_size))
        ):
            raise RuntimeError(
                "MATE GroupedLinear weights are not a complete contiguous parameter run"
            )
        out_params.extend(reversed(run_params))
        out_indices.extend(reversed(run_indices))
        changed_groups += 1
        position = end
    return out_params, out_indices, changed_groups


def _install_megatron_param_buffer_layout() -> None:
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    if getattr(_ParamAndGradBuffer, "_mate_grouped_gemm_installed", False):
        return
    original_init = _ParamAndGradBuffer.__init__

    @functools.wraps(original_init)
    def patched_init(
        self,
        ddp_config,
        param_dtype,
        grad_dtype,
        params,
        data_parallel_group,
        bucket_size,
        param_to_name,
        gradient_scaling_factor,
        param_indices,
        nccl_ub,
        pg_collection=None,
    ):
        params, param_indices, changed_groups = _reorder_marked_param_groups(
            params, param_indices
        )
        if changed_groups:
            _log(f"DDP parameter-buffer layout packed_groups={changed_groups}")
        return original_init(
            self,
            ddp_config,
            param_dtype,
            grad_dtype,
            params,
            data_parallel_group,
            bucket_size,
            param_to_name,
            gradient_scaling_factor,
            param_indices,
            nccl_ub,
            pg_collection,
        )

    _ParamAndGradBuffer.__init__ = patched_init
    _ParamAndGradBuffer._mate_grouped_gemm_installed = True


def _main_grads_ready(weights: Sequence[torch.Tensor]) -> bool:
    return all(
        isinstance(getattr(weight, "main_grad", None), torch.Tensor)
        and weight.main_grad.dtype == torch.float32
        and weight.main_grad.device == weight.device
        and weight.main_grad.shape == weight.shape
        and weight.main_grad.is_contiguous()
        for weight in weights
    )


def _unsupported_reason(module, inp, m_splits, fine_grained_offload) -> str | None:
    weights = _weights(module)
    checks = (
        (not module.fp8 and not module.fp8_calibration, "fp8"),
        (not module.use_bias and not module.return_bias, "bias"),
        (not module.gemm_bias_unfused_add, "unfused_bias"),
        (not fine_grained_offload, "offload"),
        (module.fuse_wgrad_accumulation, "fused_wgrad"),
        (inp.device.type == "musa", "device"),
        (inp.dtype == torch.bfloat16, "dtype"),
        (inp.is_contiguous(), "input_layout"),
        (isinstance(m_splits, (list, tuple)), "split_type"),
        (len(m_splits) == module.num_gemms, "split_count"),
        (all(type(value) is int and value >= 0 for value in m_splits), "split_values"),
        (sum(m_splits) == inp.reshape(-1, inp.shape[-1]).shape[0], "split_sum"),
        (_is_packed(weights), "packed_weights"),
        (all(weight.dtype == torch.bfloat16 for weight in weights), "weight_dtype"),
        (_main_grads_ready(weights), "main_grad"),
    )
    for passed, reason in checks:
        if not passed:
            return reason
    return None


def _accumulate_main_grad(weights, is_first_microbatch):
    flags = [getattr(weight, "grad_added_to_main_grad", None) for weight in weights]
    if all(flag is not None for flag in flags):
        if len({bool(flag) for flag in flags}) != 1:
            raise RuntimeError("Grouped expert weights disagree on main_grad state")
        return bool(flags[0])
    if is_first_microbatch is not None:
        return not is_first_microbatch
    return True


def _mark_main_grad_added(weights) -> None:
    for weight in weights:
        if hasattr(weight, "grad_added_to_main_grad"):
            weight.grad_added_to_main_grad = True


class _MateGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp,
        counts_device,
        host_splits,
        is_first_microbatch,
        wgrad_store,
        *weights,
    ):
        mate_gemm = load_mate_gemm()
        in_features = weights[0].shape[-1]
        out_features = weights[0].shape[0]
        flat_inp = inp.reshape(-1, in_features).contiguous()
        packed_weights = weights[0].as_strided(
            (len(weights), out_features, in_features),
            (out_features * in_features, in_features, 1),
        )
        out = torch.empty(
            (flat_inp.shape[0], out_features),
            dtype=flat_inp.dtype,
            device=flat_inp.device,
        )
        with torch.autograd.profiler.record_function("MATEGroupedLinear::fprop"):
            mate_gemm.ragged_m_moe_gemm_16bit(
                flat_inp,
                packed_weights,
                counts_device,
                out,
                gemm_mode="per_expert",
                major_a_mode="K",
                major_b_mode="K",
                backend=MATE_GEMM_BACKEND,
            )

        ctx.inp_shape = inp.shape
        ctx.host_splits = host_splits
        ctx.is_first_microbatch = is_first_microbatch
        ctx.wgrad_store = wgrad_store
        ctx.save_for_backward(flat_inp, counts_device, *weights)
        return out.reshape(*inp.shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output):
        from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
        from transformer_engine.pytorch.module.base import (
            _2X_ACC_WGRAD,
            get_multi_stream_cublas_workspace,
        )

        mate_gemm = load_mate_gemm()
        flat_inp, counts_device, *weights = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        num_experts = len(weights)
        out_features, in_features = weights[0].shape
        packed_weights = weights[0].as_strided(
            (num_experts, out_features, in_features),
            (out_features * in_features, in_features, 1),
        )

        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = torch.empty_like(flat_inp)
            with torch.autograd.profiler.record_function("MATEGroupedLinear::dgrad"):
                mate_gemm.ragged_m_moe_gemm_16bit(
                    grad_output,
                    packed_weights,
                    counts_device,
                    dgrad,
                    gemm_mode="per_expert",
                    major_a_mode="K",
                    major_b_mode="N",
                    backend=MATE_GEMM_BACKEND,
                )
            dgrad = dgrad.reshape(ctx.inp_shape)

        if any(ctx.needs_input_grad[5:]):
            input_mats = list(torch.split(flat_inp, ctx.host_splits))
            grad_output_mats = list(torch.split(grad_output, ctx.host_splits))
            wgrad_outputs = [weight.main_grad for weight in weights]
            accumulate = _accumulate_main_grad(weights, ctx.is_first_microbatch)
            grouped_gemm_wgrad = functools.partial(
                general_grouped_gemm,
                out_dtype=flat_inp.dtype,
                workspaces=get_multi_stream_cublas_workspace(),
                layout="NT",
                grad=True,
                m_splits=list(ctx.host_splits),
                use_split_accumulator=_2X_ACC_WGRAD,
                accumulate=accumulate,
            )
            if ctx.wgrad_store is not None and ctx.wgrad_store.delay_wgrad_compute():
                ctx.wgrad_store.put(
                    [input_mats, grad_output_mats, wgrad_outputs], grouped_gemm_wgrad
                )
            else:
                grouped_gemm_wgrad(input_mats, grad_output_mats, wgrad_outputs)
            _mark_main_grad_added(weights)

        return (dgrad, None, None, None, None, *([None] * num_experts))


def install_mate_grouped_gemm() -> None:
    """Replace supported TE GroupedLinear fprop/dgrad with the pinned MATE path."""
    from transformer_engine.pytorch.module.grouped_linear import GroupedLinear

    if getattr(GroupedLinear, "_mate_grouped_gemm_installed", False):
        return
    load_mate_gemm()
    original_init = GroupedLinear.__init__
    original_apply = GroupedLinear._apply
    original_forward = GroupedLinear.forward

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _pack_weights_before_ddp(self)

    @functools.wraps(original_apply)
    def patched_apply(self, fn, recurse=True):
        result = original_apply(self, fn, recurse=recurse)
        _pack_weights_before_ddp(self)
        return result

    @functools.wraps(original_forward)
    def patched_forward(
        self,
        inp,
        m_splits,
        is_first_microbatch=None,
        fine_grained_offload=False,
    ):
        unsupported_reason = _unsupported_reason(
            self, inp, m_splits, fine_grained_offload
        )
        if unsupported_reason is not None:
            raise RuntimeError(
                f"MATE GroupedLinear unsupported: {unsupported_reason}"
            )

        host_splits = tuple(m_splits)
        counts_device = torch.tensor(host_splits, dtype=torch.int32, device=inp.device)
        if not getattr(self, "_mate_active_logged", False):
            _log(
                f"active experts={self.num_gemms} input={tuple(inp.shape)} "
                f"backend={MATE_GEMM_BACKEND} "
                f"delayed_wgrad={self.wgrad_store.delay_wgrad_compute()}"
            )
            self._mate_active_logged = True
        with self.prepare_forward(inp, num_gemms=self.num_gemms) as prepared_inp:
            return _MateGroupedLinear.apply(
                prepared_inp,
                counts_device,
                host_splits,
                is_first_microbatch,
                self.wgrad_store,
                *_weights(self),
            )

    GroupedLinear.__init__ = patched_init
    GroupedLinear._apply = patched_apply
    GroupedLinear.forward = patched_forward
    GroupedLinear._mate_grouped_gemm_installed = True
    _install_megatron_param_buffer_layout()
    _log("installed: MATE fprop/dgrad + TE wgrad")
