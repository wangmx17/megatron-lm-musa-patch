# ACE FC1 routed-wgrad overlap

## Scope

This opt-in path overlaps routed FC1 weight-gradient computation with the
backward DeepEP ACE combine. FC1 dW runs after combine submission and before
the ACE completion wait; FC2 dW runs after the wait. Transformer Engine still
computes both gradients into their original FP32 `main_grad` buffers, and MATE
continues to own only routed fprop and input-gradient GEMMs.

Enable the validated stack on every rank with:

```bash
export ENABLE_DEEPEP=1
export USE_DEEPEP_ACE=1
export MUON_DP1_LOW_MEMORY=1
export DEEPEP_ACE_COMPACT_NVL_BUFFER=1
export ENABLE_ACE_FC1_WGRAD_OVERLAP=1
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
```

The overlap adapter is restricted to the tested PP1, BF16, flex/DeepEP,
expert-TP1/expert-DP1 contract without recompute, offload, bias, shared-expert
overlap, global delayed wgrad, or asynchronous gradient reduction. These gates
protect the single in-flight delayed-wgrad queues and FP32 accumulation order;
they do not change the distributed topology.

This change applies directly on top of the MATE GroupedLinear and memory-gate
baseline. At 64K, both A/B arms use Megatron's existing
`MUON_DP1_LOW_MEMORY=1` path. It processes each expert parameter's
Newton--Schulz input before moving to the next parameter instead of retaining
model-wide BF16 inputs. The path preserves momentum, Newton--Schulz, scaling,
and parameter-update math. No Muon implementation is changed by this PR.

The base PR's `DEEPEP_ACE_COMPACT_NVL_BUFFER=1` reduces only the legacy metadata
allocation; ACE payloads remain in their registered windows. It is default-off
and fail-fast bound to the audited `deep-ep==1.1.0+0bd46de`, EP8/160-expert
layout. This PR does not change that sizing implementation.

## Validation boundary

The acceptance gate is an 8-device, 64K same-source A/B with
TP2/PP1/CP4/EP8/DP1, MBS1/GBS16, BF16, MATE 0.2.7, Muon DP1 low-memory enabled,
and recompute/profiler disabled. Both arms enable ACE and the base PR's compact
NVL gate; only `ENABLE_ACE_FC1_WGRAD_OVERLAP` changes from 0 to 1. Report steps
6--30 only after both arms finish all 30 steps without skip, NaN, OOM, or
native/distributed errors. This isolates the scheduling benefit from ACE and
the memory gates.

## Rebase gate result

On worker31009, a reverse-order overlap-on then overlap-off 30-step A/B used
the contract above on base commit `ecb60f1`. Both arms completed 30/30 with
exit code 0 and no skipped, NaN, OOM, DeepEP, or MCCL errors. For steps 6--30:

- overlap off: 48.902636 s/step and 278.144 TFLOP/s/GPU;
- overlap on: 48.068260 s/step and 282.976 TFLOP/s/GPU;
- step time improved by 1.706198% and throughput by 1.737230%;
- paired overlap-on-minus-off time was -834.376 ms/step with a descriptive
  95% t interval of [-873.895, -794.857] ms. All 25 paired steady steps were
  faster with overlap enabled.

Maximum relative differences across the paired steady steps were 1.592e-5 for
loss and 0.639% for gradient norm. Transformer Engine still computes the same
FC1 and FC2 gradients into their original FP32 `main_grad` buffers; these
small BF16/distributed scheduling differences are a numerical-continuity
check, not proof of bitwise identity. Maximum allocator watermarks were
71947.44/73282 MiB allocated/reserved with overlap off and
72077.93/73662 MiB with overlap on.

This is a bounded single-node runtime and numerical-continuity result. It does
not establish long-run convergence, checkpoint round-trip, or physical
multi-node behavior.
