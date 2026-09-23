# MiniCPM5 64K memory gates

This change adds two default-off memory gates for the MiniCPM5 launcher. It is
stacked directly on the MATE GroupedLinear baseline and does not enable routed
weight-gradient overlap.

## Enable

```bash
export ENABLE_DEEPEP=1
export MUON_DP1_LOW_MEMORY=1
export DEEPEP_ACE_COMPACT_NVL_BUFFER=1
```

`ENABLE_DEEPEP=1` is the existing launcher switch for the ACE CopyEngine path;
the flex dispatcher already uses the DeepEP backend when this switch is zero.

When the Muon flag is enabled, the launcher exposes the existing compatible
Megatron `Muon` DP1 low-memory path. This PR does not replace the optimizer or
change its update implementation. In expert DP1, Megatron prepares, consumes,
and releases one BF16 Newton--Schulz input at a time instead of retaining all
inputs and identity-gather staging together. Parameters, FP32 momentum, update
equations, optimizer groups, and model-parallel groups are unchanged.

The dependency is pinned to the MiniCPM5 Megatron source snapshot
`80f9d50832e4f6b4836bb68344ea88656a9e6565`; its tested `muon.py` SHA256 is
`d4429129cf3837cbc68c43198f64019d95c5fc4b8885a4765b8fef3139e1b8ac`.

The compact NVL flag only changes the legacy metadata allocation passed to the
ACE DeepEP buffer. ACE payload windows and token capacity are unchanged. The
calculation is fail-fast bound to `deep-ep==1.1.0+0bd46de`; other builds retain
the legacy path while the compact flag is disabled, and are rejected if the
flag is enabled. For EP8 with 160 experts, the metadata payload is 896 bytes and
the implementation deliberately uses a 1 MiB aligned floor instead of the
legacy 352,784,896-byte hint. Other rank/expert geometries fail fast until their
native metadata layout is separately audited.

## Validation contract

Acceptance requires all of the following on the MATE baseline:

1. Megatron's Muon component test must show parameter and momentum equality
   (`rtol=0`, `atol=0`), including proof that the low-memory arm issues no
   identity all-gather;
2. compact-buffer unit tests and a 64K ACE capacity run;
3. a same-contract no-profiler performance A/B in which only these memory gates
   differ;
4. no skipped iterations, NaN/Inf, asynchronous MUSA/MCCL/DeepEP errors, or
   performance regression.

Run the component gates with the compatible Megatron source and this patch:

```bash
python3 test/test_ace_buffer_sizing.py
python3 -m unittest \
  Megatron-LM/tests/unit_tests/test_muon_optimizer.py
```

The end-to-end A/B must separately record the SHA256 of the actually deployed
Muon source. The source pin does not replace that runtime identity check.
