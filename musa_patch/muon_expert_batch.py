# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Opt-in independent-expert Muon batching for the compatible Megatron Muon API.

The base optimizer owns momentum, distributed reconstruction and update scaling.
Only full, unsharded 2-D expert NS inputs are grouped by shape/dtype/device.
"""
import torch
import torch.distributed as dist
from megatron.core.optimizer.muon import (
    Muon as BaseMuon,
    normalize_range,
    _zeropower_via_newtonschulz5_batched_impl,
)


class MuonExpertBatch(BaseMuon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        required = ("batch_ns_requested", "batch_ns_max_b", "dp1_low_memory_active",
                    "_prepare_muon_input", "_compute_muon_update", "_apply_muon_update")
        if any(not hasattr(self, name) for name in required):
            raise RuntimeError("TE expert batching requires the compatible Megatron Muon API")
        if self.batch_ns_max_b < 1:
            raise ValueError("MUON_BATCH_NS_MAX_B must be positive")
        self.te_expert_batch_ns_observed = set()

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

        if self.dp1_low_memory_active:
            self._step_muon_dp1_low_memory()
        else:
            # Legacy path: prepare every Muon parameter before distributed reconstruction.
            for group in self.param_groups:
                if not group.get("use_muon", False):
                    continue
                for param in group["params"]:
                    ns_inputs[param] = self._prepare_muon_input(
                        param, group["momentum"], group["nesterov"]
                    )

        if self.distributed_mode and not self.dp1_low_memory_active:
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

        if not self.dp1_low_memory_active:
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
