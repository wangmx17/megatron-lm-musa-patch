# Ordinary DeepEP FC1 weight-gradient overlap (opt-in)

This adapter does not use ACE or PR #28's compact ACE buffers. It defers the
existing Transformer Engine expert weight-gradient GEMMs; it does not replace
their arithmetic or FP32 `main_grad` accumulation.

## Scheduling change

The original `FusedDispatch.backward` submits DeepEP combine and immediately
waits for completion. Expert FC1/FC2 weight gradients have already been computed
by that point. The adapter instead uses TE's native delayed-weight-gradient
stores:

1. Compute expert input gradients and enqueue FC2/FC1 weight-gradient work.
2. Submit ordinary DeepEP backward combine asynchronously.
3. Drain FC1 weight gradients while the communication stream runs.
4. Wait for the combine result before returning input/router gradients.
5. Drain FC2 weight gradients and verify both queues are empty.

FC1 dW uses saved local expert inputs/output gradients, not the combine result,
so it can execute before the completion wait. Input contiguity and probability
casts precede the event handed to the communication stream. Queue entries retain
the tensors required for native dW until they are drained.

This moves work across a wait; it does not claim shared-expert overlap, forward
overlap, or ACE chunk-level overlap. Retaining delayed gradient inputs can raise
peak memory, so a successful short run is not sufficient evidence of stability.

## Scope and activation

The launcher defaults this feature to **off**. Add these variables to the usual
validated MiniCPM5 launch environment:

```bash
ENABLE_DEEPEP=1 USE_DEEPEP_ACE=0 ENABLE_DEEPEP_FC1_WGRAD_OVERLAP=1 \
  bash examples/minicpm5/run_16a3b.sh
```

Supported configuration is checked at initialization: Flex/DeepEP, PP1,
expert-TP1, expert-DP1, BF16 trainable expert weights, bias disabled, fused
gradient accumulation, no recomputation, no CPU offload, no FP8/FP4, no shared
expert overlap, no global delayed wgrad, and no asynchronous gradient reduction.
The expert groups are not the dense optimizer's DP×CP group. The installed MUSA
TE must provide the native `WeightGradStore` and `GroupedLinear.backward_dw`
interfaces. Unsupported configurations fail instead of silently falling back.

Disable with `ENABLE_DEEPEP_FC1_WGRAD_OVERLAP=0`; no original model files need
restoring because the patch is process-local.

## Validation — 2026-09-29

Real `minicpm5_real_part00071_text_document`, TP2/PP1/CP4/EP8, 64K sequence,
MBS1/GBS16, no recomputation, 16 MiB MCCL buffer, DeepEP SMS56, MATE v0.2.7-e5d73e9.

Six-step runs both completed with exit code zero, zero skipped/NaN iterations:

| Case | Mean iteration 2–6 |
|---|---:|
| Ordinary DeepEP, overlap off | 46.89782 s |
| Ordinary DeepEP, overlap on | 46.70510 s |

The initial 0.4109% reduction was followed by matched twenty-step runs, both
complete with exit code zero and no skipped/NaN iterations or logged errors:

| Case | Mean iteration 2–20 | Mean iteration 6–20 | Largest sampled device memory |
|---|---:|---:|---:|
| Overlap off | 46.8820421 s | 46.8314533 s | 81,096 MiB |
| Overlap on | 46.6837316 s | 46.6508333 s | 81,128 MiB |

Iteration 2–20 time decreases by **0.4230%** (0.19831 s); 18 of 19 paired
iterations are faster. This is bounded short-run evidence, not a claim about
long training or other distributed layouts. Memory is sampled every five
seconds, not an exact allocator peak; both cases have little remaining headroom.
Maximum relative differences in logged loss and gradient norm are 0.001655%
and 0.24983%, respectively. Those scalar checks do not establish full gradient
equivalence; native tensor comparisons are tracked separately. The feature
remains opt-in.

Candidate iteration-4 trace proves 1.34727 s of GEMM/DeepEP device-interval
overlap. FC1 dW overlaps communication; FC2 dW does not. The older B0 trace
shows zero GEMM/DeepEP overlap. Profiler stack settings differ, so trace wall
times are not used to estimate the speedup.

Evidence root on worker31009:
`/mbzz_ssd/wangmx/lc/experiments/minicpm5_pr2830_20260929_1305`

- `results/noace_wgrad_off6`, `results/noace_wgrad_on6_v2`: performance logs.
- `results/noace_wgrad_off20`, `results/noace_wgrad_on20`: extended validation.
- `results/noace_wgrad_trace6/trace_summary.json`: interval analysis.
- `profiler_result/noace_wgrad_trace6/iteration_4/rank0.1790675626308.pt.trace.json`:
  candidate device trace.

CPU mock tests verify scheduling and rejection guards, not GPU numerics:

```bash
python -m unittest discover -s test -p test_deepep_wgrad_schedule.py -v
```

Run `test/probe_deepep_wgrad_numerics.py` on an idle MUSA GPU using the training
environment/PYTHONPATH for full local eager/deferred gradient comparisons. This
checks empty experts and accumulated microbatches, but does not replace the
distributed training and communication-lifetime tests.

The native probe passed with exit code zero: all **126 tensor comparisons had
maximum absolute difference zero**, covering output, input gradient and all
20 FC1/20 FC2 expert weight gradients for three backward passes (initial,
accumulated, and cleared/reused). Log: `results/noace_wgrad_numerics_v4.log`.
The standalone harness initializes a one-rank Gloo group for TE workspace
queries and runs autograd on its initialized host thread. Earlier harness
attempts failed due to missing thread device context / default process group;
they did not reach a valid numerical comparison. This harness setting is not
applied to the distributed training runs above.
