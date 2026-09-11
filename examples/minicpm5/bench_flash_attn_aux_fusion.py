#!/usr/bin/env python3
"""Correctness and latency gate for the THD LSE auxiliary fusion."""

import argparse
import json
import os
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seq_len <= 0 or args.seq_len % 2:
        raise ValueError("--seq-len must be a positive even number")

    import musa_patch
    from musa_patch import flash_attn_cp_compat
    import transformer_engine.pytorch.attention as te_attention

    flash_attn_cp_compat.install_te_thd_aux_fusion(te_attention)
    correction = te_attention.tex.thd_second_half_lse_correction

    torch.manual_seed(1234)
    half = args.seq_len // 2
    initial_cpu = torch.randn(1, args.heads, args.seq_len, dtype=torch.float32)
    update_cpu = torch.randn(1, args.heads, half, dtype=torch.float32)
    expected = initial_cpu.clone()
    expected[..., half:] = torch.logaddexp(
        initial_cpu[..., half:].to(torch.float64), update_cpu.to(torch.float64)
    ).to(torch.float32)

    lse = initial_cpu.musa()
    lse_per_step = update_cpu.musa()
    cu_seqlens = torch.tensor([0, args.seq_len], dtype=torch.int32, device="musa")

    correction(lse, lse_per_step, cu_seqlens, False)
    torch.musa.synchronize()
    actual = lse.cpu()
    max_abs = (actual - expected).abs().max().item()
    max_rel = ((actual - expected).abs() / expected.abs().clamp_min(1e-6)).max().item()
    if not torch.allclose(actual, expected, rtol=2e-6, atol=2e-6, equal_nan=True):
        raise AssertionError(f"numerical mismatch: max_abs={max_abs}, max_rel={max_rel}")

    for _ in range(args.warmup):
        correction(lse, lse_per_step, cu_seqlens, False)
    torch.musa.synchronize()

    torch.musa.reset_peak_memory_stats()
    allocated_before = torch.musa.memory_allocated()
    start = time.perf_counter()
    for _ in range(args.iters):
        correction(lse, lse_per_step, cu_seqlens, False)
    torch.musa.synchronize()
    elapsed = time.perf_counter() - start
    peak_extra = torch.musa.max_memory_allocated() - allocated_before

    print(
        json.dumps(
            {
                "fusion_enabled": os.getenv("MUSA_FA_AUX_FUSION", "0") == "1",
                "shape": list(lse.shape),
                "per_step_shape": list(lse_per_step.shape),
                "warmup": args.warmup,
                "iterations": args.iters,
                "mean_us": elapsed * 1e6 / args.iters,
                "max_abs": max_abs,
                "max_rel": max_rel,
                "peak_extra_bytes": peak_extra,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
