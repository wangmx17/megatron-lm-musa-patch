# MUSA device-side metadata with a cached fallback for packed THD RoPE

## Motivation

The packed THD RoPE path derives per-sequence lengths from a MUSA
`cu_seqlens` tensor and converts them to a Python list. Repeating that work in
every Transformer layer causes repeated device-to-host copies and stream
synchronization. A CPU metadata cache avoids most repeated reads, but each
microbatch still performs one read and the RoPE path still splits and
concatenates tensors per sequence.

This change keeps `cu_seqlens` on MUSA. A Triton kernel maps every local packed
token to its context-parallel position and gathers a contiguous frequency
tensor. Native `torch.rope` then processes the complete local THD tensor in one
call. The native RoPE forward and backward kernels are unchanged.

The combined implementation retains the bounded CPU metadata cache as the
fallback for the fused and unfused THD paths. The MUSA fast path is enabled by
default for supported tensors. Set `MUSA_THD_ROPE_DEVICE_METADATA=0` before
importing `musa_patch` to use the split implementation with cached CPU sequence
lengths. Unsupported layouts automatically select the same cached fallback.

## Correctness coverage

`test/test_thd_rope_device.py` compares the device path with the existing
per-sequence reference for:

- CP sizes 1, 2, and 4, including every CP rank;
- three packed variable-length sequences;
- 1 and 16 attention heads;
- interleaved and non-interleaved RoPE;
- full rotary dimensions, with an explicit partial-dimension fallback check;
- BF16 forward results and input gradients.

All 28 forward/backward cases have exact zero maximum difference on 8 x MTT S5000. The test also
checks that CPU tensors do not select the MUSA path.

## Isolated training result

The direct comparison used the same real indexed data, random seed and old
optimization stack on 8 x MTT S5000. Both variants completed 10/10 steps with
BF16, THD/span attention, TP2 PP1 CP4 EP8 DP1, GBS16 and sequence length 65536.
Profiler runs were separate from timing runs.

| No-profiler result, steps 2-10 | CPU metadata cache | MUSA device metadata |
|---|---:|---:|
| Mean iteration time | 48.720633 s | 48.592589 s |
| Difference | - | -0.128044 s (-0.262814%) |
| Peak allocated, all ranks | 69910.313 MiB | 69922.919 MiB |
| Peak reserved, all ranks | 74118 MiB | 74536 MiB |

Both runs exited with code 0 and had no skipped or NaN iterations, OOM, or
hang. The largest per-step relative differences were 0.00150234% for loss and
0.01389495% for gradient norm. Gradient norm comparison is not a proof of
elementwise full-model gradient equivalence.

This is a single short sequential comparison. The 0.263% result is an observed
small improvement, not proof of statistical significance. It also includes
the removal of per-sequence Python `split`/loop/`cat`, so it must not be
attributed only to moving the remaining CPU metadata read onto MUSA.

## Trace evidence

Each trace is rank 0, training step 4, from an independent 10-step run.

| Sampled training step | CPU metadata cache | MUSA device metadata |
|---|---:|---:|
| RoPE/cache calls | 896 | 896 |
| `.tolist()` inside RoPE | 16 | 0 |
| `musaMemcpyAsync` inside RoPE | 16 | 0 |
| `musaStreamSynchronize` inside RoPE | 16 | 0 |
| `aten::split` inside RoPE | 896 | 0 |
| `aten::cat` inside RoPE | 1792 | 0 |
| frequency-gather kernels | 0 | 896 |
| native RoPE forward kernels | 896 | 896 |
| native RoPE backward kernels | 896 | 896 |
| Device activity span | 49.733060 s | 49.628682 s |
| Device busy interval union | 46.332586 s | 46.221579 s |
| Device gaps over 100 us | 1.213074 s | 1.303150 s |

The CPU cache made 896 queries but only 16 device reads in the sampled step,
one for each microbatch. Across the complete 10-step run, every rank made 8960
queries and 160 reads, a 98.2143% hit rate. This verifies that the comparison
used a working CPU cache rather than the uncached baseline.

The MUSA path removed those remaining 16 reads and added 896 frequency-gather
kernels totaling about 20.992 ms. The native RoPE kernel counts remained
unchanged. Device gaps over 100 us did not improve, so inclusive host timings
and gap categories should not be added together or treated as iteration-time
savings.

## Scope and limitations

The optimized path requires MUSA tensors, contiguous int32 `cu_seqlens`,
`freqs` shaped `[S, 1, 1, R]`, a non-empty THD tensor, full rotary dimensions,
and frequencies that do not require gradients. Packed sequence lengths must
obey Megatron's CP divisibility and symmetric two-chunk layout contract.

Unsupported inputs retain the split implementation and use the bounded CPU
metadata cache instead of repeating device-to-host sequence-length reads in
every layer. Current native MUSA `torch.rope` rejects partial rotary dimensions,
so they deliberately do not select the device fast path. Long-run convergence,
additional dtypes, malformed sequence boundaries and other CP layouts remain
outside this validation.
