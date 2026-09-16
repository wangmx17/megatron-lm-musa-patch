# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
# if os.getenv("ACCELERATOR_BACKEND", "musa") == "musa":
if os.getenv("ACCELERATOR_BACKEND") == "musa":
    import musa_patch
else:
    import cuda_patch
# Optional one-shot op shape dump (DUMP_OP_SHAPES=1; rank0 / once per tag).
try:
    from op_shape_dump import install_op_shape_dump_hooks

    install_op_shape_dump_hooks()
except Exception as _op_shape_dump_exc:  # pragma: no cover
    print(f"[op_shape_dump] install skipped: {_op_shape_dump_exc!r}", flush=True)
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)


stimer = StragglerDetector()

# Optional iter spike diagnostics (DUMP_LOSS_DIAG=1):
# - wrap LanguageModule.compute_language_model_loss for max|logits|
# - dump per-token CE stats in loss_func
_DUMP_LOSS_DIAG = int(os.getenv("DUMP_LOSS_DIAG", "0"))
_DUMP_MECH_PROBE = int(os.getenv("DUMP_MECH_PROBE", "0"))
_LOSS_DIAG_INSTALLED = False
_MECH_PROBE_INSTALLED = False
_MECH_TRAIN_STEP_WRAPPED = False
_MECH_HIDDEN_STATS = {}
_CYCLE13 = (109186, 7739, 66, 42, 386, 21806, 1318, 83007, 9935, 3856, 4746, 35, 242)


def _mech_log(msg: str) -> None:
    print(msg, flush=True)
    save_dir = os.getenv("SAVE_DIR", "")
    if save_dir and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "mech_probe.txt"), "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def _mech_should_log() -> bool:
    if not torch.distributed.is_initialized():
        return True
    try:
        return (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
            and mpu.get_data_parallel_rank() == 0
        )
    except Exception:
        return torch.distributed.get_rank() == 0


def install_mech_probes_on_model(model) -> None:
    """Attach hidden / router / attn / emb probes (DUMP_MECH_PROBE=1)."""
    global _MECH_PROBE_INSTALLED
    if _MECH_PROBE_INSTALLED or not _DUMP_MECH_PROBE:
        return
    # model may be list/DDP wrapped
    mods = model if isinstance(model, (list, tuple)) else [model]
    targets = []
    for m in mods:
        core = m.module if hasattr(m, "module") else m
        targets.append(core)

    def _hidden_hook(name):
        def hook(_mod, _inp, out):
            try:
                t = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(t):
                    return
                x = t.detach().float()
                rms = torch.sqrt(torch.mean(x * x))
                mx = x.abs().max()
                _MECH_HIDDEN_STATS[name] = (float(rms.item()), float(mx.item()))
            except Exception:
                pass
        return hook

    def _softmax_entropy_hook(name):
        def hook(_mod, _inp, out):
            try:
                t = out[0] if isinstance(out, (tuple, list)) else out
                if not torch.is_tensor(t):
                    return
                p = t.detach().float()
                # Expect probs along last dim.
                ent = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1).mean()
                top = p.max(dim=-1).values.mean()
                _MECH_HIDDEN_STATS[f"attn::{name}"] = (float(ent.item()), float(top.item()))
            except Exception:
                pass
        return hook

    hooked_layers = 0
    hooked_routers = 0
    hooked_attn = 0
    for core in targets:
        # Unwrap nested DDP/Float16 module wrappers.
        walk = core
        for _ in range(4):
            if hasattr(walk, "module") and walk.module is not None:
                walk = walk.module
            else:
                break
        core = walk

        # Decoder layers (direct attr or via named_modules)
        layers = None
        if hasattr(core, "decoder") and hasattr(core.decoder, "layers"):
            layers = core.decoder.layers
        elif hasattr(core, "language_model") and hasattr(core.language_model, "encoder"):
            layers = getattr(core.language_model.encoder, "layers", None)
        if layers is None:
            layer_mods = []
            for name, mod in core.named_modules():
                if name.endswith(".decoder.layers") or name == "decoder.layers":
                    layers = mod
                    break
                # Collect TransformerLayer-like modules under decoder.layers.N
            if layers is None:
                indexed = []
                for name, mod in core.named_modules():
                    parts = name.split(".")
                    if len(parts) >= 2 and parts[-2] == "layers" and parts[-1].isdigit():
                        if "decoder" in name:
                            indexed.append((int(parts[-1]), mod))
                if indexed:
                    indexed.sort()
                    # Fake a list-like access via dict
                    class _LayerList(list):
                        pass
                    layers = _LayerList(m for _, m in indexed)

        if layers is not None:
            n = len(layers)
            for idx in sorted({0, n // 2, n - 1}):
                if 0 <= idx < n:
                    layers[idx].register_forward_hook(_hidden_hook(f"layer{idx}"))
                    hooked_layers += 1
                    attn = getattr(layers[idx], "self_attention", None) or getattr(
                        layers[idx], "attention", None
                    )
                    if attn is not None:
                        attn.register_forward_hook(_hidden_hook(f"attn_out{idx}"))

        # MoE routers: best-effort name match
        for name, mod in core.named_modules():
            cls = mod.__class__.__name__.lower()
            if "router" in cls or name.endswith(".router"):
                def _router_hook(n):
                    def hook(_mod, _inp, out):
                        try:
                            logits = out[0] if isinstance(out, (tuple, list)) else out
                            if not torch.is_tensor(logits):
                                return
                            p = torch.softmax(logits.detach().float(), dim=-1)
                            # entropy
                            ent = -(p * (p.clamp_min(1e-12).log())).sum(dim=-1).mean()
                            top = p.max(dim=-1).values.mean()
                            _MECH_HIDDEN_STATS[f"router::{n}"] = (float(ent.item()), float(top.item()))
                        except Exception:
                            pass
                    return hook
                mod.register_forward_hook(_router_hook(name))
                hooked_routers += 1
                if hooked_routers >= 3:
                    break

        # Softmax / fused softmax (non-flash paths) for true attention entropy.
        for name, mod in core.named_modules():
            cls = mod.__class__.__name__.lower()
            if "softmax" in cls:
                mod.register_forward_hook(_softmax_entropy_hook(name.split(".")[-1]))
                hooked_attn += 1
                if hooked_attn >= 3:
                    break

        # Embedding weight norms for cycle tokens (local shard) + emb grad RMS.
        emb = getattr(core, "embedding", None)
        if emb is not None and hasattr(emb, "word_embeddings"):
            weight = emb.word_embeddings.weight

            def _emb_fwd_hook(_mod, _inp, _out, w=weight):
                try:
                    with torch.no_grad():
                        # Best-effort under TP: local row index == global id only if unsharded.
                        norms = []
                        for tid in _CYCLE13:
                            if tid < w.size(0):
                                norms.append(float(w[tid].float().norm().item()))
                        if norms:
                            _MECH_HIDDEN_STATS["cycle_emb_norm"] = (
                                float(sum(norms) / len(norms)),
                                float(max(norms)),
                            )
                except Exception:
                    pass

            emb.register_forward_hook(_emb_fwd_hook)

            def _emb_grad_hook(grad):
                try:
                    if grad is None:
                        return grad
                    g = grad.detach().float()
                    rms = torch.sqrt(torch.mean(g * g))
                    mx = g.abs().max()
                    _MECH_HIDDEN_STATS["emb_grad"] = (float(rms.item()), float(mx.item()))
                    # Fraction of grad mass on local rows whose index is in CYCLE13 (approx under TP).
                    cycle_set = set(_CYCLE13)
                    row_sq = (g * g).sum(dim=-1)
                    total = float(row_sq.sum().item()) + 1e-12
                    local_cycle = [i for i in range(g.size(0)) if i in cycle_set]
                    if local_cycle:
                        frac = float(row_sq[local_cycle].sum().item()) / total
                        _MECH_HIDDEN_STATS["cycle_grad_frac"] = (frac, float(len(local_cycle)))
                except Exception:
                    pass
                return grad

            if weight.requires_grad:
                weight.register_hook(_emb_grad_hook)

    _install_adam_step_probe()
    _MECH_PROBE_INSTALLED = True
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        print(
            f"[mech_probe] installed hooks layers={hooked_layers} routers={hooked_routers} "
            f"softmax={hooked_attn} (flash path: attn_out RMS only)",
            flush=True,
        )


def _dump_adam_effective_step(optimizer) -> None:
    """Log ||lr * m / (sqrt(v)+eps)|| over Adam state (pre-clip update proxy uses current grads if no state)."""
    if not _DUMP_MECH_PROBE or not _mech_should_log():
        return
    try:
        args = get_args()
        lr = float(args.lr) if args.lr is not None else 0.0
        # Prefer live param-group lr.
        try:
            groups = getattr(optimizer, "param_groups", None)
            if groups:
                lr = float(groups[0].get("lr", lr))
        except Exception:
            pass
        eps = float(os.getenv("ADAM_EPS", "1e-8"))

        # Unwrap chained / distributed optimizers to torch Adam-like.
        opt_list = []
        if hasattr(optimizer, "chained_optimizers"):
            for o in optimizer.chained_optimizers:
                inner = getattr(o, "optimizer", o)
                opt_list.append(inner)
        elif hasattr(optimizer, "optimizer"):
            opt_list.append(optimizer.optimizer)
        else:
            opt_list.append(optimizer)

        sq_sum = 0.0
        n_tensors = 0
        n_with_state = 0
        for opt in opt_list:
            state = getattr(opt, "state", None)
            groups = getattr(opt, "param_groups", None)
            if state is None or groups is None:
                continue
            for group in groups:
                group_lr = float(group.get("lr", lr))
                group_eps = float(group.get("eps", eps))
                for p in group["params"]:
                    if p is None:
                        continue
                    st = state.get(p, None)
                    if st is None or "exp_avg" not in st or "exp_avg_sq" not in st:
                        continue
                    m = st["exp_avg"].detach().float()
                    v = st["exp_avg_sq"].detach().float()
                    step = group_lr * m / (torch.sqrt(v) + group_eps)
                    sq_sum += float((step * step).sum().item())
                    n_tensors += 1
                    n_with_state += 1
        if n_with_state == 0:
            _mech_log(f"[mech_probe][adam] lr={lr:.6g} effective_step=NA (no adam state yet)")
            return
        eff_norm = sq_sum ** 0.5
        _mech_log(
            f"[mech_probe][adam] lr={lr:.6g} eff_step_norm={eff_norm:.6g} "
            f"n_tensors={n_tensors}"
        )
    except Exception as exc:
        print(f"[mech_probe] adam dump failed: {exc!r}", flush=True)


_FORCE_OPT_APPLIED = False


def _force_optimizer_hyperparams(optimizer) -> None:
    """ckpt resume restores param_group max_lr/eps; FORCE_* overrides after load."""
    global _FORCE_OPT_APPLIED
    if _FORCE_OPT_APPLIED:
        return
    force_lr = os.getenv("FORCE_MAX_LR", "").strip()
    force_eps = os.getenv("FORCE_ADAM_EPS", "").strip()
    force_min = os.getenv("FORCE_MIN_LR", "").strip()
    if not force_lr and not force_eps:
        _FORCE_OPT_APPLIED = True
        return
    try:
        n = 0
        opt_list = []
        if hasattr(optimizer, "chained_optimizers"):
            opt_list.extend(optimizer.chained_optimizers)
        else:
            opt_list.append(optimizer)
        for opt in opt_list:
            groups = getattr(opt, "param_groups", None)
            if groups is None and hasattr(opt, "optimizer"):
                groups = getattr(opt.optimizer, "param_groups", None)
            if not groups:
                continue
            for g in groups:
                if force_lr:
                    g["max_lr"] = float(force_lr)
                    g["lr"] = float(force_lr)
                if force_min:
                    g["min_lr"] = float(force_min)
                if force_eps:
                    g["eps"] = float(force_eps)
                n += 1
        # Also patch scheduler max_lr if present on args path later via train_step.
        try:
            args = get_args()
            if force_lr:
                args.lr = float(force_lr)
            if force_min:
                args.min_lr = float(force_min)
        except Exception:
            pass
        _FORCE_OPT_APPLIED = True
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            print(
                f"[force_opt] applied groups={n} FORCE_MAX_LR={force_lr or '-'} "
                f"FORCE_MIN_LR={force_min or '-'} FORCE_ADAM_EPS={force_eps or '-'}",
                flush=True,
            )
    except Exception as exc:
        print(f"[force_opt] failed: {exc!r}", flush=True)


def _install_train_step_wrappers() -> None:
    """Wrap megatron train_step for Adam dump and/or FORCE_* hyperparams."""
    global _MECH_TRAIN_STEP_WRAPPED
    need = _DUMP_MECH_PROBE or bool(os.getenv("FORCE_MAX_LR", "").strip()) or bool(
        os.getenv("FORCE_ADAM_EPS", "").strip()
    )
    if _MECH_TRAIN_STEP_WRAPPED or not need:
        return
    try:
        import megatron.training.training as tr

        if getattr(tr.train_step, "_mech_probe_wrapped", False):
            _MECH_TRAIN_STEP_WRAPPED = True
            return
        _orig = tr.train_step

        def _wrapped(forward_step_func, data_iterator, model, optimizer, *args, **kwargs):
            _force_optimizer_hyperparams(optimizer)
            try:
                it = int(getattr(get_args(), "curr_iteration", -1))
            except Exception:
                it = -1
            if _DUMP_MECH_PROBE:
                _dump_adam_effective_step(optimizer)
                if _mech_should_log():
                    _mech_log(f"[mech_probe][iter_tag] curr_iteration={it}")
            return _orig(forward_step_func, data_iterator, model, optimizer, *args, **kwargs)

        _wrapped._mech_probe_wrapped = True
        tr.train_step = _wrapped
        _MECH_TRAIN_STEP_WRAPPED = True
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            print("[mech_probe] wrapped megatron.training.training.train_step", flush=True)
    except Exception as exc:
        print(f"[mech_probe] train_step wrap failed: {exc!r}", flush=True)


def _install_adam_step_probe() -> None:
    _install_train_step_wrappers()


def _flush_mech_probe_stats(tag: str = "") -> None:
    if not _DUMP_MECH_PROBE or not _MECH_HIDDEN_STATS:
        return
    try:
        if not _mech_should_log():
            return
        parts = [f"[mech_probe]{tag}"]
        for k in sorted(_MECH_HIDDEN_STATS.keys()):
            a, b = _MECH_HIDDEN_STATS[k]
            if k.startswith("router::") or k.startswith("attn::"):
                parts.append(f"{k}:ent={a:.4g},top1p={b:.4g}")
            elif k == "emb_grad":
                parts.append(f"emb_grad_rms={a:.4g},max={b:.4g}")
            elif k == "cycle_grad_frac":
                parts.append(f"cycle_grad_frac={a:.4g},n_local={b:.0f}")
            elif k == "cycle_emb_norm":
                parts.append(f"cycle_emb_norm_mean={a:.4g},max={b:.4g}")
            else:
                parts.append(f"{k}:rms={a:.4g},max={b:.4g}")
        _mech_log(" ".join(parts))
        _MECH_HIDDEN_STATS.clear()
    except Exception as exc:
        print(f"[mech_probe] flush failed: {exc!r}", flush=True)


def _install_loss_diag_hooks() -> None:
    """Monkeypatch CE entry to log logit magnitudes once per microbatch (last rank)."""
    global _LOSS_DIAG_INSTALLED
    if _LOSS_DIAG_INSTALLED or not _DUMP_LOSS_DIAG:
        return
    from megatron.core.models.common.language_module.language_module import LanguageModule
    import megatron.core.tensor_parallel as tensor_parallel

    _orig = LanguageModule.compute_language_model_loss

    def _wrapped(self, labels, logits):
        try:
            with torch.no_grad():
                local_abs = logits.detach().float().abs()
                local_max = local_abs.max()
                local_mean = local_abs.mean()
                # TP shards vocab dim; reduce max/mean across TP for a global view.
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        local_max, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group()
                    )
                    mean_buf = torch.stack([local_mean, local_mean.new_tensor(float(local_abs.numel()))])
                    torch.distributed.all_reduce(
                        mean_buf, op=torch.distributed.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group()
                    )
                    global_mean = mean_buf[0] / mean_buf[1].clamp_min(1.0)
                else:
                    global_mean = local_mean
                # FP32-logits CE probe on this TP shard (same labels); compare mean CE vs bf16 path.
                labels_t = labels.transpose(0, 1).contiguous()
                ce_bf16 = tensor_parallel.vocab_parallel_cross_entropy(logits.detach(), labels_t)
                ce_fp32 = tensor_parallel.vocab_parallel_cross_entropy(logits.detach().float(), labels_t)
                ce_bf16_mean = ce_bf16.float().mean()
                ce_fp32_mean = ce_fp32.float().mean()
                if torch.distributed.is_initialized():
                    for t in (ce_bf16_mean, ce_fp32_mean):
                        torch.distributed.all_reduce(
                            t, op=torch.distributed.ReduceOp.AVG, group=mpu.get_tensor_model_parallel_group()
                        )
                msg = (
                    f"[loss_diag] logits dtype={logits.dtype} shape={tuple(logits.shape)} "
                    f"max_abs={float(local_max):.6g} mean_abs={float(global_mean):.6g} "
                    f"ce_mean_bf16_logits={float(ce_bf16_mean):.6g} "
                    f"ce_mean_fp32_logits={float(ce_fp32_mean):.6g} "
                    f"ce_mean_delta={float(ce_fp32_mean - ce_bf16_mean):.6g}"
                )
                # Prefer last rank so it lands next to training progress lines.
                if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
                    if mpu.get_context_parallel_rank() == 0:
                        print(msg, flush=True)
                save_dir = os.getenv("SAVE_DIR", "")
                if save_dir and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                    os.makedirs(save_dir, exist_ok=True)
                    with open(os.path.join(save_dir, "loss_diag_logits.txt"), "a", encoding="utf-8") as f:
                        f.write(msg + "\n")
        except Exception as exc:  # pragma: no cover
            print(f"[loss_diag] logits probe failed: {exc!r}", flush=True)
        return _orig(self, labels, logits)

    LanguageModule.compute_language_model_loss = _wrapped
    _LOSS_DIAG_INSTALLED = True
    print("[loss_diag] installed compute_language_model_loss probe", flush=True)


_install_loss_diag_hooks()


def _dump_per_token_ce_stats(losses: torch.Tensor, loss_mask: torch.Tensor) -> None:
    """Dump per-token CE distribution for the current microbatch (masked tokens only)."""
    if not _DUMP_LOSS_DIAG:
        return
    try:
        with torch.no_grad():
            flat = losses.detach().float().view(-1)
            mask = loss_mask.detach().float().view(-1) > 0
            if not bool(mask.any()):
                return
            vals = flat[mask]
            total = float(vals.sum().item())
            n = int(vals.numel())
            # Top-1% contribution
            k = max(1, n // 100)
            topk = torch.topk(vals, k).values
            top1pct_frac = float(topk.sum().item()) / max(total, 1e-12)
            qs = torch.quantile(vals, torch.tensor([0.5, 0.9, 0.99, 0.999], device=vals.device))
            msg = (
                f"[loss_diag] per_token_ce n={n} mean={float(vals.mean()):.6g} "
                f"max={float(vals.max()):.6g} "
                f"p50={float(qs[0]):.6g} p90={float(qs[1]):.6g} "
                f"p99={float(qs[2]):.6g} p999={float(qs[3]):.6g} "
                f"top1pct_loss_frac={top1pct_frac:.6g}"
            )
            if mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
                if mpu.get_data_parallel_rank() == 0:
                    print(msg, flush=True)
            save_dir = os.getenv("SAVE_DIR", "")
            if save_dir and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                os.makedirs(save_dir, exist_ok=True)
                with open(os.path.join(save_dir, "loss_diag_ce.txt"), "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
    except Exception as exc:  # pragma: no cover
        print(f"[loss_diag] ce stats failed: {exc!r}", flush=True)


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    config: Optional["TransformerConfig"] = None,
    pg_collection=None,
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        vp_stage (int, optional): Virtual pipeline stage index (Megatron-LM v0.16 build_model injects this).
        config (TransformerConfig, optional): v0.16 build_model injects the transformer config;
            when None we build it from args (legacy behaviour).
        pg_collection (ProcessGroupCollection, optional): v0.16 process groups, forwarded to GPTModel.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

    print_rank_0('building GPT model ...')
    # v0.16 build_model injects `config`; only build from args when not provided.
    if config is None:
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    _dump_per_token_ce_stats(losses, loss_mask)
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(rerun_state_machine.is_spiky_loss, threshold=SPIKY_LOSS_PERC),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    if not int(os.getenv("NO_LOSS_REDUCE", 0)): #TODO:(huang.huang) will influence the loss reported Now!
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

        if int(os.getenv("USE_EPX", 0)):
            from musa_patch.fault_tolerance_epx.epx_sync_tensor import epx_sync_tensor_across_replicas
            epx_sync_tensor_across_replicas(reporting_loss, opts=torch.distributed.ReduceOp.AVG, assemble=False)

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()
    # Ensure FORCE_* / Adam probe wrappers install even when DUMP_MECH_PROBE=0.
    _install_train_step_wrappers()
    install_mech_probes_on_model(model)

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        batch = get_batch(data_iterator)
        tokens = batch['tokens']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']
        # v0.16 get_batch_on_this_cp_rank always adds packed_seq_params (needed for CP/span-attn).
        packed_seq_params = batch.get('packed_seq_params')
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels, packed_seq_params=packed_seq_params)
    _flush_mech_probe_stats(tag="")

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    try:
        pretrain(
            train_valid_test_datasets_provider,
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        )
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
