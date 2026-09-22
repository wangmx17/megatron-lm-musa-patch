# MATE GroupedLinear for MiniCPM5

## Design

This branch uses MATE 0.2.7 `ragged_m_moe_gemm_16bit` as the default BF16
routed-expert GroupedLinear implementation. There is no runtime enable flag:
importing `musa_patch` loads and installs the adapter before Transformer Engine
binds its native modules.

The ownership boundary is intentionally narrow:

- MATE computes routed-expert forward and input-gradient GEMMs;
- Transformer Engine continues to compute weight gradients and accumulates them
  directly into FP32 `main_grad`;
- routing, expert order, TP/PP/CP/EP/DP groups, optimizer math, and checkpoint
  parameter identities are unchanged;
- CPU affinity, deferred route counts, MATE ragged-K wgrad, ACE, and activation
  offload are not part of this change.

MATE requires the 20 expert weights of each GroupedLinear to be one contiguous
expert-major allocation. The adapter packs weights after construction or device
conversion, marks each expert parameter, and compensates for Megatron's reverse
DDP-buffer walk while applying the same permutation to `param_indices`.
Unsupported dtype, device, bias, FP8, offload, split, storage, or `main_grad`
contracts fail fast instead of silently falling back to Transformer Engine.

MATE 0.2.7 must be importable by every rank. The MiniCPM5 launcher now forwards
the caller's existing `PYTHONPATH` through its nested SSH launch; this locates
Python dependencies but is not an optimization switch. The adapter pins MATE
source version `0.2.7` and backend `mubin` in code.

Install the validated MATE source and expose it to every rank:

```bash
git clone https://github.com/MooreThreads/mate.git --recursive
git -C mate checkout e5d73e914fcb30aab915a12fae146d9d26357908
git -C mate submodule update --init --recursive
export PYTHONPATH="/path/to/mate:${PYTHONPATH:-}"
```

## Validation

The validation used upstream baseline
`9d4cac85c9c7fc7a21435431e0b4e7b912ef5788` on one worker31009 node with
8 MTT S5000 GPUs:

- MiniCPM5 16A3B, BF16, sequence length 65536;
- TP/PP/CP/EP/DP = 2/1/4/8/1;
- micro batch 1, global batch 16, 16 microbatches;
- branch-default optimization stack, no activation recompute, no profiler;
- random initialization, identical seed/data/model contract, `NO_SAVE=1`;
- installed Transformer Engine lacks native THD-LSE-fp32, so both arms used the
  repository's `MUSA_TE_THD_LSE_FP32=disable` compatibility path;
- MATE source was v0.2.7 commit
  `e5d73e914fcb30aab915a12fae146d9d26357908`.

Correctness and integration gates:

- 5 CPU helper tests passed;
- 16 real route-shape probes covering balanced/p50/p95/max, FC1/FC2, and
  fprop/dgrad passed; maximum relative-L2 was `5.026e-6`;
- the synchronized API probe was `1.46x--2.48x` faster than independent
  per-expert torch matmuls; this is a diagnostic reference, not TE or training
  throughput;
- an 8-rank 3-step smoke completed 3/3 with every rank on the MATE `mubin`
  path;
- both no-profiler 30-step arms completed 30/30 with zero skipped iterations,
  zero NaN iterations, and no fatal signature.

## 64K performance

Steady-state statistics use steps 6--30, 25 samples per arm, in
baseline-then-MATE execution order:

| Metric | Upstream baseline | MATE |
|---|---:|---:|
| Mean step time | 48.358084 s | 48.193480 s |
| Median step time | 48.3637 s | 48.1899 s |
| Population stdev | 0.118805 s | 0.080364 s |
| Mean TFLOP/s/GPU | 281.288 | 282.248 |
| Max allocated | 71648.52 MiB | 71645.40 MiB |
| Max reserved | 74926 MiB | 74910 MiB |

MATE reduced mean step time by `164.604 ms` (`0.340386%`) and increased
throughput by `0.341548%`. The paired MATE-minus-baseline mean had a descriptive
normal-approximation 95% interval of `[-208.320, -120.888] ms`.

The maximum loss relative difference was `2.669e-5`; the maximum grad-norm
relative difference was `0.4621%`. Allocator peaks changed by only
`-3.12 MiB` allocated and `-16 MiB` reserved, so this is a performance change,
not a memory optimization. MATE's first step was `65.4245 s` versus the
baseline's `63.8616 s`, reflecting extra first-use loading outside the steady
window.

This is one ordered 30-step random-initialized A/B pair. It does not establish
long-run convergence, reverse-order reproducibility, or checkpoint and optimizer
state round-trip compatibility.
