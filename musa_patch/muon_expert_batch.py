# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Opt-in independent-expert Muon batching for the stock Megatron Muon API.

This adapter intentionally owns the small compatibility layer needed by PR11 so
that Megatron-LM does not need an out-of-tree ``muon.py`` modification.  The
base optimizer still owns its parameter groups and distributed metadata.  Only
full, unsharded 2-D expert NS inputs are grouped by shape/dtype/device.
"""
import os

import torch
import torch.distributed as dist
from megatron.core.optimizer.muon import (
    Muon as BaseMuon,
    adjust_lr_wd_for_muon,
    normalize_range,
    zeropower_via_newtonschulz5,
)


def _read_env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 0/1, false/true, no/yes, or off/on; got {value!r}"
    )


def _read_env_positive_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return parsed


def _zeropower_via_newtonschulz5_batched_impl(
    matrices, steps, coefficient_type
):
    """Apply independent Newton--Schulz iterations to equally shaped matrices."""
    if matrices.dim() != 3:
        raise ValueError(
            "batched Newton--Schulz expects [batch, rows, cols], "
            f"got shape={tuple(matrices.shape)}"
        )
    # Keep the coefficient source in Megatron so this adapter follows the exact
    # simple/quintic/polar-express selection used by the base optimizer.
    from megatron.core.optimizer.muon import _COEFFICIENT_SETS

    coefficient_sets = _COEFFICIENT_SETS[coefficient_type]
    x = matrices
    transposed = matrices.size(-2) > matrices.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    # Normalizing each matrix independently is required for numerical
    # equivalence with the original per-parameter loop.
    x = x / (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True) + 1e-7)
    for iteration in range(steps):
        a, b, c = coefficient_sets[iteration % len(coefficient_sets)]
        a_matrix = x @ x.transpose(-2, -1)
        b_matrix = b * a_matrix + c * a_matrix @ a_matrix
        x = a * x + b_matrix @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x


class MuonExpertBatch(BaseMuon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_ns_requested = _read_env_flag("MUON_BATCH_NS", default=False)
        self.batch_ns_max_b = _read_env_positive_int(
            "MUON_BATCH_NS_MAX_B", default=32
        )
        if _read_env_flag("MUON_DP1_LOW_MEMORY", default=False):
            raise RuntimeError(
                "MUON_DP1_LOW_MEMORY is not supported together with "
                "MUON_TE_EXPERT_BATCH_NS; the validated 1000-step stack uses "
                "MUON_DP1_LOW_MEMORY=0"
            )
        self.dp1_low_memory_active = False
        self.dist_world_size = 1
        self.tp_world_size = 1
        self.tp_rank = 0
        self.te_expert_batch_ns_observed = set()

    def enable_distributed_mode(
        self, global_buffer_sizes, dist_group, tp_group, dist_metas
    ):
        """Preserve base setup and retain the topology needed by the adapter."""
        super().enable_distributed_mode(
            global_buffer_sizes, dist_group, tp_group, dist_metas
        )
        self.dist_world_size = dist.get_world_size(dist_group)
        self.tp_world_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

    def _prepare_muon_input(self, param, momentum, nesterov):
        """Update FP32 momentum and materialize one BF16 NS input."""
        grad = param.grad
        if grad is None:
            raise RuntimeError("Muon parameter is missing its gradient")
        if not self.distributed_mode and grad.dim() != 2:
            raise ValueError(
                "non-distributed Muon parameters must be 2-D, "
                f"got shape={tuple(grad.shape)}"
            )
        state = self.state[param]
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(grad)
        momentum_buffer = state["exp_avg"]
        momentum_buffer.mul_(momentum).add_(grad)
        ns_input = (
            grad.add(momentum_buffer, alpha=momentum)
            if nesterov
            else momentum_buffer
        )
        return ns_input.bfloat16()

    def _compute_muon_update(self, param, ns_input, ns_steps):
        """Preserve the base Muon TP, QKV and distributed-shard semantics."""
        tp_split_dim = -1
        dist_meta = None
        if self.distributed_mode:
            dist_meta = self.dist_metas[param]
            tp_split_dim = dist_meta.tp_split_dim

        if tp_split_dim != -1:
            ns_input_shards = [
                torch.empty_like(ns_input) for _ in range(self.tp_world_size)
            ]
            dist.all_gather(ns_input_shards, ns_input, self.tp_group)
            ns_input = torch.cat(ns_input_shards, dim=tp_split_dim)

        scale_shape = ns_input.shape
        if self.muon_split_qkv and getattr(param, "is_qkv", False):
            num_query_groups = ns_input.shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                ns_input.view(
                    num_query_groups, sum(self.qkv_split_shapes), -1
                ),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [
                grad.reshape(-1, ns_input.shape[-1]) for grad in qkv_grads
            ]
            qkv_grads = [
                zeropower_via_newtonschulz5(
                    grad,
                    steps=ns_steps,
                    coefficient_type=self.muon_coefficient_type,
                ).view(num_query_groups, -1, ns_input.shape[-1])
                for grad in qkv_grads
            ]
            update = torch.cat(qkv_grads, dim=1).view(ns_input.shape)
        else:
            update = zeropower_via_newtonschulz5(
                ns_input,
                steps=ns_steps,
                coefficient_type=self.muon_coefficient_type,
            )

        if tp_split_dim != -1:
            update = update.chunk(self.tp_world_size, dim=tp_split_dim)[self.tp_rank]

        if self.distributed_mode:
            local_range = normalize_range(
                dist_meta.local_range, dist_meta.global_range[0]
            )
            update = update.reshape(-1)[local_range[0] : local_range[1]]
        return update, scale_shape

    def _apply_muon_update(self, param, update, scale_shape, group):
        """Apply the same weight decay and Muon learning-rate scaling as Megatron."""
        lr = group["lr"]
        adjusted_lr = adjust_lr_wd_for_muon(
            lr, group["matched_adamw_rms"], scale_shape
        )
        param.data.mul_(1 - lr * group["weight_decay"])
        param.data.add_(update, alpha=-adjusted_lr)

    def _can_batch_te_expert_param(self, param, ns_input, group):
        """Return true only for the exact DP1 expert path validated by this experiment."""
        if not self.batch_ns_requested or not group.get("is_expert_parallel", False):
            return False
        if self.muon_split_qkv and getattr(param, "is_qkv", False):
            return False
        if ns_input.dim() != 2:
            return False
        if not self.distributed_mode:
            return True
        dist_meta = self.dist_metas[param]
        return (
            self.dist_world_size == 1
            and dist_meta.tp_split_dim == -1
            and tuple(ns_input.shape) == tuple(dist_meta.shape)
            and dist_meta.local_range == dist_meta.global_range
        )

    def _apply_te_expert_batches(self, group, params, ns_inputs):
        """Batch independent, equally shaped TE expert matrices without coupling norms."""
        batches = {}
        fallback = []
        for param in params:
            ns_input = ns_inputs[param]
            if not self._can_batch_te_expert_param(param, ns_input, group):
                fallback.append(param)
                continue
            key = (tuple(ns_input.shape), ns_input.dtype, ns_input.device)
            batches.setdefault(key, []).append(param)

        for key, same_shape_params in batches.items():
            for start in range(0, len(same_shape_params), self.batch_ns_max_b):
                chunk = same_shape_params[start : start + self.batch_ns_max_b]
                if len(chunk) == 1:
                    fallback.extend(chunk)
                    continue
                stacked = torch.stack([ns_inputs[param] for param in chunk], dim=0)
                updates = _zeropower_via_newtonschulz5_batched_impl(
                    stacked,
                    steps=group["ns_steps"],
                    coefficient_type=self.muon_coefficient_type,
                )
                marker = (key[0], len(chunk))
                if marker not in self.te_expert_batch_ns_observed:
                    print(
                        f"[MUON_TE_EXPERT_BATCH_NS] shape={key[0]} "
                        f"batch={len(chunk)} max_batch={self.batch_ns_max_b}",
                        flush=True,
                    )
                    self.te_expert_batch_ns_observed.add(marker)
                for param, update in zip(chunk, updates.unbind(0)):
                    apply_update = update.reshape(-1) if self.distributed_mode else update
                    self._apply_muon_update(param, apply_update, key[0], group)
                del stacked, updates

        for param in fallback:
            update, scale_shape = self._compute_muon_update(
                param, ns_inputs[param], group["ns_steps"]
            )
            self._apply_muon_update(param, update, scale_shape, group)

    def step(self):
        dtype = torch.bfloat16
        ns_inputs = {}

        # Prepare every Muon parameter before distributed reconstruction, matching
        # the original optimizer order used by the validated 1000-step run.
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue
            for param in group["params"]:
                ns_inputs[param] = self._prepare_muon_input(
                    param, group["momentum"], group["nesterov"]
                )

        if self.distributed_mode:
            if ns_inputs:
                device = next(iter(ns_inputs)).device
            else:
                device = torch.device("cuda", torch.cuda.current_device())
            # initialize buffers
            ns_input_local_buffers = [
                [ torch.empty((local_buffer_size), device=device, dtype=dtype)
                    for local_buffer_size in local_bucket_sizes ]
                for local_bucket_sizes in self.local_buffer_sizes
            ]
            ns_input_global_buffers = [
                [ torch.empty((global_buffer_size), device=device, dtype=dtype)
                    for (global_buffer_size, bucket_offset) in global_bucket_sizes ]
                for global_bucket_sizes in self.global_buffer_sizes
            ]
            # fill ns input data to local buffer
            for param, ns_input in ns_inputs.items():
                dist_meta = self.dist_metas[param]
                ns_input_local_buffer = ns_input_local_buffers[dist_meta.buffer_idx][dist_meta.bucket_idx]
                local_buffer_range = self.local_buffer_ranges[dist_meta.buffer_idx][dist_meta.bucket_idx]
                local_range = normalize_range(dist_meta.local_range, local_buffer_range[0])
                ns_input_local_buffer[local_range[0]:local_range[1]].copy_(ns_input.view(-1))
            # all gather buffers
            for ns_input_global_buffer, ns_input_local_buffer in zip(ns_input_global_buffers, ns_input_local_buffers):
                for ns_input_global_bucket, ns_input_local_bucket in zip(ns_input_global_buffer, ns_input_local_buffer):
                    dist.all_gather_into_tensor(ns_input_global_bucket, ns_input_local_bucket, group=self.dist_group)
            # overwrite ns input
            for p in ns_inputs.keys():
                dist_meta = self.dist_metas[p]
                ns_input_global_buffer = ns_input_global_buffers[dist_meta.buffer_idx][dist_meta.bucket_idx]
                global_range = dist_meta.global_range
                offset = self.global_buffer_sizes[dist_meta.buffer_idx][dist_meta.bucket_idx][1]
                ns_inputs[p] = ns_input_global_buffer[
                    global_range[0] - offset : global_range[1] - offset
                ].view(dist_meta.shape)

        for group in self.param_groups:
            if not group.get('use_muon', False):
                continue
            group['step'] = group.get('step', 0) + 1
            if self.batch_ns_requested and group.get("is_expert_parallel", False):
                self._apply_te_expert_batches(group, group["params"], ns_inputs)
            else:
                for param in group["params"]:
                    update, scale_shape = self._compute_muon_update(
                        param, ns_inputs[param], group["ns_steps"]
                    )
                    self._apply_muon_update(param, update, scale_shape, group)

        # use adam for other params
        for group in self.param_groups:
            if group.get('use_muon', False):
                continue
            # init step
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1
            step = group['step']
            params = group["params"]
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(g)
                    state['exp_avg_sq'] = torch.zeros_like(g)
                buf1 = state['exp_avg']
                buf2 = state['exp_avg_sq']
                buf1.lerp_(g, 1-beta1)
                buf2.lerp_(g.square(), 1-beta2)
                g = buf1 / (eps + buf2.sqrt())
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr/scale)
