"""Bounded diagnostic: two MoEs, consecutive F/B without per-micro CPU checks.

This exercises the routed dW/backward ACE combine schedule and is not a throughput benchmark.
Final finite checks detect gross corruption; they are not a numerical oracle.
"""
import copy
import json
import os
from pathlib import Path

import pretrain_minicpm5_musa as training_entry
import torch
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.initialize import initialize_megatron


def main():
    initialize_megatron(args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    args = get_args()
    config = core_transformer_config_from_args(args)
    assert not config.moe_shared_expert_overlap
    assert os.environ.get("USE_DEEPEP_ACE") == "1"
    assert config.tensor_model_parallel_size == 2
    spec = training_entry.get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts, moe_grouped_gemm=args.moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
    ).submodules.mlp.submodules
    rank = torch.distributed.get_rank()
    model_parallel_cuda_manual_seed(1234)
    layers = []
    for number in (2, 3):
        layer = MoELayer(copy.deepcopy(config), spec).cuda().to(config.params_dtype)
        layer.set_layer_number(number)
        layer.train()
        if config.gradient_accumulation_fusion:
            for param in layer.parameters():
                param.main_grad = torch.zeros_like(param, dtype=torch.float32)
        queue = getattr(layer.token_dispatcher._comm_manager, "wgrad_overlap", None)
        assert (queue is not None) == (os.environ["ENABLE_ACE_WGRAD_OVERLAP"] == "1")
        layers.append(layer)
    rows = args.seq_length // args.context_parallel_size // args.tensor_model_parallel_size
    g = torch.Generator(device="cpu").manual_seed(900 + rank)
    value = torch.randn(rows, 1, config.hidden_size, generator=g).cuda().to(config.params_dtype)
    upstream = torch.randn(rows, 1, config.hidden_size, generator=g).cuda().to(config.params_dtype)
    # Shape/source evidence is metadata only, before the stress loop.
    print("ACE_WGRAD_STRESS_METADATA=" + json.dumps({
        "rank": rank, "input": list(value.shape),
        "overlap": os.environ["ENABLE_ACE_WGRAD_OVERLAP"],
    }), flush=True)
    for micro in range(128):
        x = value.detach().clone().requires_grad_(True)
        y = x
        for layer in layers:
            y2, bias = layer(y)
            y = y + y2 if bias is None else y + y2 + bias
        y.backward(upstream / 128)
        # No synchronize/item/cpu/barrier/log in the loop. Let the allocator
        # and autograd exercise normal cross-micro storage and stream reuse.
    for layer in layers:
        queue = getattr(layer.token_dispatcher._comm_manager, "wgrad_overlap", None)
        if queue is not None:
            assert not queue.active and queue.completed == 128
            assert all(x.wgrad_store.context.empty() for x in queue.layers)
    torch.cuda.synchronize()
    tensors = [y.detach(), x.grad]
    for layer in layers:
        for param in layer.parameters():
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is not None:
                tensors.append(grad)
    finite = all(t is not None and bool(torch.isfinite(t).all()) for t in tensors)
    ok = torch.tensor(int(finite), device=value.device)
    torch.distributed.all_reduce(ok, op=torch.distributed.ReduceOp.MIN)
    result = {"rank": rank, "micro_count": 128, "moe_calls": 256,
              "pass": bool(ok.item()), "scope": "native-failure-and-finite-stress-only"}
    out = Path(os.environ["SAVE_DIR"]) / "stress"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
    print("ACE_WGRAD_STRESS_RESULT=" + json.dumps(result), flush=True)
    assert result["pass"], "Nonfinite result in ACE backward wgrad stress"


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()
