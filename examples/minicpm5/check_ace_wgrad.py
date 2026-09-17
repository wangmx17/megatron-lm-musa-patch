"""Distributed MoE output/gradient oracle, launched with the training CLI.

Use PRETRAIN_FILE=<this file> with the normal MiniCPM5 launcher. This builds
two isolated MoE layers, not the full model or a dataset. Both arms use ACE;
only routed backward wgrad scheduling differs; both use no shared overlap. No optimizer updates are made.
"""

import copy
import json
import os
from pathlib import Path

if os.environ.get("ENABLE_ACE_WGRAD_OVERLAP") != "1":
    raise RuntimeError(
        "Set ENABLE_ACE_WGRAD_OVERLAP=1 before importing the MUSA patches; "
        "the component test creates its ACE-only control after installation"
    )

# Import the same accelerator patches and model-spec providers as training.
import pretrain_minicpm5_musa as training_entry
import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.initialize import initialize_megatron


def snapshot_gradients(model):
    result = {}
    for name, param in model.named_parameters():
        grad = param.grad
        if getattr(param, "grad_added_to_main_grad", False) or grad is None:
            grad = getattr(param, "main_grad", grad)
        result[name] = None if grad is None else grad.detach().float().cpu().clone()
    return result


def run(model, state, value, upstream):
    model.load_state_dict(state)
    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        if hasattr(param, "main_grad"):
            param.main_grad.zero_()
        if hasattr(param, "grad_added_to_main_grad"):
            param.grad_added_to_main_grad = False
    model_parallel_cuda_manual_seed(1234)
    x = value.detach().clone().requires_grad_(True)
    expert_counts = []

    def capture_counts(module, inputs):
        expert_counts.append(inputs[1].detach().clone())

    handle = model.experts.register_forward_pre_hook(capture_counts)
    try:
        for micro in range(3):
            x = value.detach().clone().requires_grad_(True)
            y, bias = model(x)
            if bias is not None:
                y = y + bias
            y.backward(upstream / 3)
            queue = getattr(model.token_dispatcher._comm_manager, "wgrad_overlap", None)
            if queue is not None:
                assert not queue.active
                assert all(layer.wgrad_store.context.empty() for layer in queue.layers)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    return (
        y.detach().float().cpu(), x.grad.detach().float().cpu(),
        snapshot_gradients(model), [counts.cpu() for counts in expert_counts],
    )


def compare(name, expected, actual):
    if expected is None or actual is None:
        return {"name": name, "pass": expected is None and actual is None, "missing": True}
    if expected.shape != actual.shape:
        return {"name": name, "pass": False, "reason": "shape_mismatch"}
    if not bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()):
        return {"name": name, "pass": False, "reason": "nonfinite"}
    delta = actual - expected
    ref_norm = expected.norm().item()
    max_ref = expected.abs().max().item() if expected.numel() else 0.0
    max_error = delta.abs().max().item() if delta.numel() else 0.0
    rel_l2 = delta.norm().item() / max(ref_norm, 1e-12)
    # Declared before execution. Do not loosen after a failure: investigate first.
    # Absolute floor handles identically zero/small gradients, relative criteria
    # cover BF16 reduction-order differences; the repeated baseline is reported too.
    ok = (
        delta.norm().item() <= 0.02 * ref_norm + 1e-6
        and max_error <= 0.05 * max_ref + 1e-6
    )
    return {"name": name, "pass": ok, "relative_l2": rel_l2, "max_abs": max_error}


def main():
    if os.environ.get("USE_DEEPEP_ACE") != "1":
        raise RuntimeError("This component comparison requires ACE on for both arms")
    initialize_megatron(args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    args = get_args()
    config = core_transformer_config_from_args(args)
    if config.moe_token_dispatcher_type != "flex" or not config.sequence_parallel:
        raise RuntimeError("Expected the current Flex + sequence-parallel training configuration")
    spec = training_entry.get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
    ).submodules.mlp.submodules
    layers = []
    for overlap in (False, True):
        cfg = copy.deepcopy(config)
        cfg.moe_shared_expert_overlap = False
        os.environ["ENABLE_ACE_WGRAD_OVERLAP"] = str(int(overlap))
        model_parallel_cuda_manual_seed(1234)
        layer = MoELayer(cfg, spec).cuda().to(dtype=config.params_dtype)
        layer.set_layer_number(2)
        layer.train()
        if cfg.gradient_accumulation_fusion:
            for param in layer.parameters():
                param.main_grad = torch.zeros_like(param, dtype=torch.float32)
        layers.append(layer)
    assert not hasattr(layers[0].token_dispatcher._comm_manager, "wgrad_overlap")
    assert layers[1].token_dispatcher._comm_manager.wgrad_overlap is not None
    state = copy.deepcopy(layers[0].state_dict())
    rank = torch.distributed.get_rank()
    local_rows = args.seq_length // args.context_parallel_size // args.tensor_model_parallel_size
    # Largest allocation first, followed by smaller shapes and another full reuse.
    cases = [("full", local_rows), ("small", 16), ("zero_experts", 32), ("reuse", local_rows)]
    reports = []
    passed = True
    for label, rows in cases:
        case_state = copy.deepcopy(state)
        if label == "zero_experts":
            # Equal logits deterministically select a small expert subset, leaving
            # other experts empty. Exercise actual routing and its gradients.
            case_state["router.weight"].zero_()
        generator = torch.Generator(device="cpu").manual_seed(900 + rank + rows)
        value = torch.randn(rows, 1, config.hidden_size, generator=generator).cuda().to(config.params_dtype)
        upstream = torch.randn(rows, 1, config.hidden_size, generator=generator).cuda().to(config.params_dtype)
        baseline = run(layers[0], case_state, value, upstream)
        repeated = run(layers[0], case_state, value, upstream)
        candidate = run(layers[1], case_state, value, upstream)
        routing_counts = {
            arm: [counts.tolist() for counts in result[3]]
            for arm, result in [("baseline", baseline), ("baseline_repeat", repeated), ("overlap", candidate)]
        }
        # Some EP ranks can receive only selected experts. Check empty-expert
        # coverage across all ranks while retaining each rank's actual counts.
        coverage = torch.tensor(
            [int(any(bool((counts == 0).any()) for counts in result[3]))
             for result in (baseline, repeated, candidate)], device=value.device,
        )
        torch.distributed.all_reduce(coverage, op=torch.distributed.ReduceOp.MAX)
        coverage_pass = label != "zero_experts" or bool(coverage.all())
        for arm, outputs in [("baseline_repeat", repeated), ("overlap", candidate)]:
            metrics = [compare("output", baseline[0], outputs[0]), compare("input_grad", baseline[1], outputs[1])]
            metrics.extend(compare(name, grad, outputs[2][name]) for name, grad in baseline[2].items())
            routing_match = len(baseline[3]) == len(outputs[3]) == 3 and all(
                torch.equal(a, b) for a, b in zip(baseline[3], outputs[3])
            )
            arm_pass = all(item["pass"] for item in metrics) and coverage_pass and routing_match
            passed &= arm_pass
            reports.append({
                "case": label, "rows": rows, "arm": arm, "pass": arm_pass,
                "metrics": metrics, "tokens_per_expert": routing_counts,
                "routing_match": routing_match, "empty_expert_coverage": coverage.tolist(),
            })
        torch.distributed.barrier()
    out = Path(os.environ["SAVE_DIR"]) / "component"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rank{rank}.json").write_text(json.dumps({"rank": rank, "pass": passed, "cases": reports}, indent=2))
    success = torch.tensor(int(passed), device=value.device)
    torch.distributed.all_reduce(success, op=torch.distributed.ReduceOp.MIN)
    if rank == 0:
        print(f"ACE_WGRAD_COMPONENT_PASS={bool(success.item())}", flush=True)
    if not success.item():
        raise AssertionError("ACE wgrad output/gradient comparison failed; inspect all rank reports")


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()
