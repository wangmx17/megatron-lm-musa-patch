#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 12 ]]; then
  echo "Usage: bash $0 WORK_HOME PATCH_HOME EXPNAME HOSTFILE DATA_PATH TP_SIZE PP_SIZE MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE MODEL_PATH RDZV_ID MASTER_PORT"
  echo "Got $# arguments."
  exit 1
fi

WORK_HOME=$1
PATCH_HOME=$2
EXPNAME=$3
HOSTFILE=$4
DATA_PATH=$5
TP_SIZE=$6
PP_SIZE=$7
MICRO_BATCH_SIZE=$8
GLOBAL_BATCH_SIZE=$9
MODEL_PATH=${10}
RDZV_ID=${11}
MASTER_PORT=${12}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_FILE=${PRETRAIN_FILE:-"${SCRIPT_DIR}/pretrain_minicpm5_musa.py"}
EP_SIZE=${EP_SIZE:-4}
CP_SIZE=${CP_SIZE:-4}
EXPERT_TP_SIZE=${EXPERT_TP_SIZE:-1}

# MiniCPM5 16A3B: 16 routed experts active per token, about 3B active parameters.
NUM_LAYERS=28
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=32
NUM_QUERY_GROUPS=2
FFN_HIDDEN_SIZE=8192
VOCAB_SIZE=130560
SEQ_LENGTH=${SEQ_LENGTH:-65536}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-${SEQ_LENGTH}}
KV_CHANNELS=128
ROTARY_BASE=${ROTARY_BASE:-5000000}
NORM_EPS=1e-6
MAKE_VOCAB_SIZE_DIVISIBLE_BY=1

NUM_EXPERTS=160
MOE_ROUTER_TOPK=16
MOE_FFN_HIDDEN_SIZE=512
SHARED_EXPERT_INTERMEDIATE_SIZE=512
MOE_LAYER_FREQ='[0]+[1]*27'
MOE_ROUTER_TOPK_SCALING_FACTOR=3.66

export LD_LIBRARY_PATH=/usr/local/musa/lib:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}
# The v2.1.7-rc3 image has a zero-byte /lib64/libmtperf_target.so ahead of
# the usable profiler library. Prefer the valid target library only for
# profiler runs so ordinary training keeps its established library order.
if [[ "${ENABLE_PROFILER:-0}" -eq 1 ]] && [[ -s /usr/lib/x86_64-linux-gnu/libmtperf_target.so ]]; then
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
  echo "Profiler library path enabled: /usr/lib/x86_64-linux-gnu"
fi
export PATH=/usr/local/musa/bin:/usr/local/musa/mccl_test:/usr/local/openmpi/bin:${PATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MTHREADS_VISIBLE_DEVICES=${MTHREADS_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export MUSA_KERNEL_TIMEOUT=${MUSA_KERNEL_TIMEOUT:-3200000}
export ACCELERATOR_BACKEND=musa
export MCCL_PROTOS=${MCCL_PROTOS:-2}
export MCCL_ALGOS=${MCCL_ALGOS:-1}
export MCCL_BUFFSIZE=${MCCL_BUFFSIZE:-16777216}
export MUSA_BLOCK_SCHEDULE_MODE=${MUSA_BLOCK_SCHEDULE_MODE:-1}
export MCCL_IB_GID_INDEX=${MCCL_IB_GID_INDEX:-3}
export MCCL_NET_SHARED_BUFFERS=${MCCL_NET_SHARED_BUFFERS:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_MUSA_ALLOC_CONF=${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}
export TORCH_MCCL_AVOID_RECORD_STREAMS=${TORCH_MCCL_AVOID_RECORD_STREAMS:-1}
# One-shot Attention/Router/MoE/dispatcher shape dump (rank0, once per tag).
export DUMP_OP_SHAPES=${DUMP_OP_SHAPES:-0}
export DUMP_OP_SHAPES_RANK=${DUMP_OP_SHAPES_RANK:-0}
if [[ -n "${SAVE_DIR:-}" ]]; then
  export DUMP_OP_SHAPES_PATH=${DUMP_OP_SHAPES_PATH:-"${SAVE_DIR}/op_shapes_rank${DUMP_OP_SHAPES_RANK}.txt"}
fi
# Iter16 spike probes (see iter16_grad_spike_problem_summary.md §7).
export DUMP_LOSS_DIAG=${DUMP_LOSS_DIAG:-0}
export DUMP_MECH_PROBE=${DUMP_MECH_PROBE:-0}
export CLIP_GRAD=${CLIP_GRAD:-1.0}
export DISABLE_MOE_FUSIONS=${DISABLE_MOE_FUSIONS:-0}
export ADAM_EPS=${ADAM_EPS:-1e-8}
# After resume, Megatron keeps ckpt param_group max_lr/eps; FORCE_* patches them.
export FORCE_MAX_LR=${FORCE_MAX_LR:-}
export FORCE_MIN_LR=${FORCE_MIN_LR:-}
export FORCE_ADAM_EPS=${FORCE_ADAM_EPS:-}

MEGATRON_PATH=${MEGATRON_PATH:-/home/Megatron-LM}
export PYTHONPATH=${MEGATRON_PATH}:${PATCH_HOME}:${PYTHONPATH:-}

for required in "${MEGATRON_PATH}" "${PATCH_HOME}" "${WORK_HOME}" "${MODEL_PATH}"; do
  [[ -d "${required}" ]] || {
    echo "Error: required directory not found: ${required}"
    exit 2
  }
done
[[ -f "${PRETRAIN_FILE}" ]] || {
  echo "Error: missing ${PRETRAIN_FILE}"
  exit 2
}
[[ -f "${HOSTFILE}" ]] || { echo "Error: HOSTFILE not found: ${HOSTFILE}"; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || {
  echo "Error: config.json not found: ${MODEL_PATH}/config.json"
  exit 2
}
if [[ ! -f "${MODEL_PATH}/tokenizer.json" && ! -f "${MODEL_PATH}/tokenizer.model" ]]; then
  echo "Error: tokenizer files not found under ${MODEL_PATH}"
  exit 2
fi

python3 - <<MUSACHECK
import sys
try:
    import torch
    import torch_musa
    available = torch.musa.is_available()
    count = torch.musa.device_count()
    print(f"MUSA check: available={available} device_count={count}", flush=True)
    sys.exit(0 if available and count >= int("${GPUS_PER_NODE:-8}") else 9)
except Exception as exc:
    print(f"MUSA check exception: {exc!r}", flush=True)
    sys.exit(9)
MUSACHECK

python3 - "${MODEL_PATH}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    config = json.load(stream)
expected = {
    "model_type": "minicpm5_moe",
    "num_hidden_layers": 28,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "intermediate_size": 8192,
    "vocab_size": 130560,
    "max_position_embeddings": 4096,
    "head_dim": 128,
    "moe_intermediate_size": 512,
    "n_routed_experts": 160,
    "n_shared_experts": 1,
    "num_experts_per_tok": 16,
    "first_k_dense_replace": 1,
    "routed_scaling_factor": 3.66,
}
mismatches = [
    f"{key}: config={config.get(key)!r}, script={value!r}"
    for key, value in expected.items()
    if config.get(key) != value
]
if mismatches:
    print("Error: MiniCPM5 16A3B config mismatch:", file=sys.stderr)
    for mismatch in mismatches:
        print("  - " + mismatch, file=sys.stderr)
    sys.exit(2)
PY

CHECKPOINT_PATH=${SAVE_DIR:-${WORK_HOME}/output/${EXPNAME}}
DATA_CACHE_PATH=${WORK_HOME}/data_cache/${EXPNAME}
LOG_PATH=${WORK_HOME}/logs/${EXPNAME}
TB_PATH=${TENSORBOARD_DIR:-${WORK_HOME}/tboard/${EXPNAME}}
mkdir -p "${CHECKPOINT_PATH}" "${DATA_CACHE_PATH}" "${LOG_PATH}" "${TB_PATH}"
cp "$0" "${LOG_PATH}/"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=$(awk '!/^#/ && NF {count++} END {print count+0}' "${HOSTFILE}")
MASTER_ADDR=$(awk '!/^#/ && NF {print $1; exit}' "${HOSTFILE}")
NODE_ADDR=$(
  ip -o -4 addr show | awk -v hostfile="${HOSTFILE}" '
    BEGIN {
      while ((getline line < hostfile) > 0) {
        if (line !~ /^#/ && line ~ /[^[:space:]]/) {
          split(line, fields, /[[:space:]]+/)
          wanted[fields[1]] = 1
        }
      }
    }
    $0 !~ /127.0.0.1/ {
      split($4, addr, "/")
      if (wanted[addr[1]]) {
        print addr[1]
        exit
      }
    }'
)
NODE_RANK=$(awk -v node_addr="${NODE_ADDR}" '!/^#/ && NF {ranks[$1]=(seen++);} END {print ranks[node_addr]}' "${HOSTFILE}")
[[ -n "${NODE_ADDR}" && -n "${NODE_RANK}" ]] || {
  echo "Error: failed to resolve NODE_ADDR/NODE_RANK from ${HOSTFILE}"
  exit 2
}

# Per-node torchrun logs under host_<IP>/....  Use the standard PyTorch module so
# this example remains self-contained inside code/Megatron-LM.
OUTPUT_LOG_ROOT="${OUTPUT_LOG_ROOT:-${WORK_HOME}/output_log}"
TORCHRUN_LOG_DIR="${OUTPUT_LOG_ROOT}/${RDZV_ID}/${EXPNAME}/host_${NODE_ADDR}"
mkdir -p "${TORCHRUN_LOG_DIR}"
TORCHRUN_BIN=(python3 -m torch.distributed.run)

DISTRIBUTED_ARGS=(
  --nproc_per_node "${GPUS_PER_NODE}"
  --nnodes "${NUM_NODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  --rdzv-id "${RDZV_ID}"
  --log_dir "${TORCHRUN_LOG_DIR}"
  --redirects "${LOG_REDIRECTS_LEVEL:-3}"
  --tee "${LOG_TEE_LEVEL:-0}"
)

MODEL_ARGS=(
  --num-layers "${NUM_LAYERS}"
  --hidden-size "${HIDDEN_SIZE}"
  --num-attention-heads "${NUM_ATTENTION_HEADS}"
  --group-query-attention
  --num-query-groups "${NUM_QUERY_GROUPS}"
  --seq-length "${SEQ_LENGTH}"
  --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
  --norm-epsilon "${NORM_EPS}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --disable-bias-linear
  --vocab-size "${VOCAB_SIZE}"
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
  --position-embedding-type rope
  --no-position-embedding
  --swiglu
  --normalization RMSNorm
  --untie-embeddings-and-output-weights
  --no-rope-fusion
  --make-vocab-size-divisible-by "${MAKE_VOCAB_SIZE_DIVISIBLE_BY}"
  --rotary-percent 1.0
  --rotary-base "${ROTARY_BASE}"
  --use-rope-scaling
  --rope-type llama
  --rope-scaling-factor 8.0
  --kv-channels "${KV_CHANNELS}"
)

MOE_ARGS=(
  --num-experts "${NUM_EXPERTS}"
  --expert-model-parallel-size "${EP_SIZE}"
  --moe-router-topk "${MOE_ROUTER_TOPK}"
  --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}"
  --moe-shared-expert-intermediate-size "${SHARED_EXPERT_INTERMEDIATE_SIZE}"
  --moe-layer-freq "${MOE_LAYER_FREQ}"
  --moe-router-load-balancing-type seq_aux_loss
  --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-0}"
  --moe-token-dispatcher-type flex
  --moe-router-score-function sigmoid
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate "${MOE_ROUTER_BIAS_UPDATE_RATE:-0}"
  --moe-router-pre-softmax
  --moe-router-topk-scaling-factor "${MOE_ROUTER_TOPK_SCALING_FACTOR}"
  --moe-router-dtype fp32
  --moe-deepep-num-sms "${MOE_DEEPEP_NUM_SMS:-20}"
)
# DISABLE_MOE_FUSIONS=1: drop permute-fusion + grouped-gemm (precision / fusion ablation).
# DISABLE_PERMUTE_FUSION=1: drop ONLY permute-fusion, keep grouped-gemm (isolate fused permute kernel).
if [[ "${DISABLE_MOE_FUSIONS:-0}" -ne 1 && "${DISABLE_PERMUTE_FUSION:-0}" -ne 1 ]]; then
  MOE_ARGS+=(--moe-permute-fusion)
fi
if [[ "${USE_GROUPED_GEMM:-1}" -eq 1 && "${DISABLE_MOE_FUSIONS:-0}" -ne 1 ]]; then
  MOE_ARGS+=(--moe-grouped-gemm)
fi

TRAINING_ARGS=(
  --seed "${SEED:-42}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --lr-warmup-iters "${LR_WARMUP_ITERS:-1}"
  --train-iters "${TRAIN_ITERS:-2}"
  --init-method-std 0.02
  --use-mcore-models
  --no-bias-dropout-fusion
  --distributed-backend nccl
  --use-distributed-optimizer
  --context-parallel-size "${CP_SIZE}"
  --expert-tensor-parallel-size "${EXPERT_TP_SIZE}"
  --no-create-attention-mask-in-dataloader
)
if [[ "${DISABLE_RECOMPUTE:-1}" -eq 1 ]]; then
  echo "Activation recompute: disabled"
else
  TRAINING_ARGS+=(
    --recompute-granularity full
    --recompute-method "${RECOMPUTE_METHOD:-block}"
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-8}"
  )
fi
# Span-based attention (THD/packed) — matches the H800-validated config; saves activation
# memory at 64K. Disable with USE_SPAN_BASED_ATTN=0.
[[ "${USE_SPAN_BASED_ATTN:-1}" -eq 1 ]] && TRAINING_ARGS+=(--use-span-based-attn)
# Optimizer selection (adam default in Megatron); use Muon per the H800 config / fork support.
TRAINING_ARGS+=(--optimizer "${OPTIMIZER:-muon}")
# Rerun engine: default Megatron validate_results; RERUN_MODE=disabled turns off the iter-0 rerun (方案一).
[[ -n "${RERUN_MODE:-}" ]] && TRAINING_ARGS+=(--rerun-mode "${RERUN_MODE}")
[[ "${USE_FLASH_ATTN:-1}" -eq 1 ]] && TRAINING_ARGS+=(--use-flash-attn)
if (( TP_SIZE > 1 )); then
  TRAINING_ARGS+=(--sequence-parallel --tp-only-amax-red)
fi

if [[ "${USE_MOCK_DATA:-0}" -eq 1 ]]; then
  DATA_ARGS=(
    --mock-data
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${MODEL_PATH}"
  )
else
  [[ -f "${DATA_PATH}.bin" && -f "${DATA_PATH}.idx" ]] || {
    echo "Error: indexed dataset not found: ${DATA_PATH}.bin/.idx"
    exit 2
  }
  DATA_ARGS=(
    --data-path "${DATA_PATH}"
    --data-cache-path "${DATA_CACHE_PATH}"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${MODEL_PATH}"
    --split 1
  )
fi

# RESUME=1: full Megatron resume (weights + optim + rng + iteration).
# Default remains --finetune (iteration reset to 0; no optim/rng), for HF/smoke starts.
LOAD_ARGS=()
if [[ -n "${LOAD_DIR:-}" && -f "${LOAD_DIR}/latest_checkpointed_iteration.txt" ]]; then
  if [[ "${RESUME:-0}" -eq 1 ]]; then
    LOAD_ARGS=(--load "${LOAD_DIR}")
    echo "Resume load from ${LOAD_DIR} (no --finetune; keep iteration/optim/rng)"
  else
    LOAD_ARGS=(--load "${LOAD_DIR}" --finetune)
  fi
elif [[ -f "${MODEL_PATH}/latest_checkpointed_iteration.txt" ]]; then
  if [[ "${RESUME:-0}" -eq 1 ]]; then
    LOAD_ARGS=(--load "${MODEL_PATH}")
    echo "Resume load from ${MODEL_PATH} (no --finetune; keep iteration/optim/rng)"
  else
    LOAD_ARGS=(--load "${MODEL_PATH}" --finetune)
  fi
elif [[ -f "${MODEL_PATH}/pytorch_model.bin" || -f "${MODEL_PATH}/model.safetensors" ]]; then
  echo "Warning: ${MODEL_PATH} contains Hugging Face weights, not a Megatron checkpoint."
  [[ "${ALLOW_RANDOM_INIT:-0}" -eq 1 ]] || {
    echo "Error: convert the checkpoint first, or set ALLOW_RANDOM_INIT=1 for a smoke test."
    exit 2
  }
  [[ "${RESUME:-0}" -eq 1 ]] && {
    echo "Error: RESUME=1 requires a Megatron checkpoint with latest_checkpointed_iteration.txt"
    exit 2
  }
fi

# CLIP_GRAD: override default 1.0 (set 0 to disable clipping for spike ablation).
REGULARIZATION_ARGS=(--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps "${ADAM_EPS:-1e-8}" --clip-grad "${CLIP_GRAD:-1.0}")
LEARNING_RATE_ARGS=(
  --lr "${LR:-2e-5}"
  --lr-decay-style "${LR_DECAY_STYLE:-WSD}"
  --min-lr "${MIN_LR:-2e-6}"
  --initial-loss-scale "${INITIAL_LOSS_SCALE:-65536}"
  --min-loss-scale "${MIN_LOSS_SCALE:-1.0}"
)
if [[ "${LR_DECAY_STYLE:-WSD}" == "WSD" ]]; then
  LEARNING_RATE_ARGS+=(
    --lr-decay-iters "${LR_DECAY_ITERS:-46000}"
    --lr-wsd-decay-iters "${LR_WSD_DECAY_ITERS:-46000}"
    --lr-wsd-decay-style "${LR_WSD_DECAY_STYLE:-exponential}"
  )
fi
# When extending train-iters on resume, override ckpt scheduler fields (e.g. wd_incr_steps).
# Megatron flag spelling is literally --override-opt_param-scheduler (underscore).
if [[ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" -eq 1 ]]; then
  LEARNING_RATE_ARGS+=(--override-opt_param-scheduler)
fi
MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
)
MIXED_PRECISION_ARGS=(
  --bf16
  --attention-softmax-in-fp32
  --no-masked-softmax-fusion
  --accumulate-allreduce-grads-in-fp32
)
LOGGING_ARGS=(
  --log-interval 1
  --log-throughput
  --eval-interval 1
  --eval-iters 0
  --save-interval "${SAVE_INTERVAL:-100000}"
  --tensorboard-dir "${TB_PATH}"
)
# NO_SAVE=1: perf ablation only, skip all checkpoint saving (avoids 193G/run on full disk).
if [[ "${NO_SAVE:-0}" -ne 1 ]]; then
  LOGGING_ARGS+=(--save "${CHECKPOINT_PATH}")
fi
TRANSFORMER_ENGINE_ARGS=(--transformer-impl "${TRANSFORMER_IMPL:-transformer_engine}")

# ---- Experimental optimization toggles (opt-ablation copy; stack via ENABLE_*) ----
# DeepEP follows DeepSeek-V3 internal summary: USE_DEEPEP_ACE + flex + enable-deepep
# + token-drop probs + enable-experimental + MCCL_CROSS_NIC.
# gradient-accumulation-fusion: baseline already True (no --no-gradient-accumulation-fusion).
OPT_ARGS=()
# 1. DeepEP + ACE (CopyEngine async). Baseline already uses flex/deepep backend;
#    this adds ACE path + deprecated moe-enable-deepep + experimental flags from DS doc.
if [[ "${ENABLE_DEEPEP:-0}" -eq 1 ]]; then
  export USE_DEEPEP_ACE="${USE_DEEPEP_ACE:-1}"
  export MCCL_CROSS_NIC="${MCCL_CROSS_NIC:-1}"
  # ACE Buffer construction (patched get_buffer in musa_patch/deepep_ace/fused_a2a_ace.py).
  # Must cover worst-card hs0 * topk = 65536 * 16.
  export DEEPEP_ACE_TOKEN_NUM="${DEEPEP_ACE_TOKEN_NUM:-65536}"
  export DEEPEP_ACE_NUM_TOPK="${DEEPEP_ACE_NUM_TOPK:-16}"
  export DEEPEP_ACE_HIDDEN="${DEEPEP_ACE_HIDDEN:-2048}"
  export DEEPEP_ACE_NUM_BUFFERS="${DEEPEP_ACE_NUM_BUFFERS:-1}"
  # DeepEP Test New.pdf runtime env. This node is RoCE (mlx5 link_layer=Ethernet).
  # mtshmem is required by DeepEP tests; low-latency IBGDA cpu/gpumem flags are NOT
  # applied here (those are for test_low_latency.py, not ACE CopyEngine).
  if [[ -d /usr/local/mtshmem/lib ]]; then
    export LD_LIBRARY_PATH="/usr/local/mtshmem/lib:${LD_LIBRARY_PATH}"
  fi
  if [[ -d /usr/local/mtshmem/bin ]]; then
    export PATH="/usr/local/mtshmem/bin:${PATH}"
  fi
  export MUSA_DEVICE_PAGE_SIZE="${MUSA_DEVICE_PAGE_SIZE:-0x1000}"
  export NVSHMEM_IB_SL="${NVSHMEM_IB_SL:-7}"
  export NVSHMEM_IB_TRAFFIC_CLASS="${NVSHMEM_IB_TRAFFIC_CLASS:-163}"
  MOE_ARGS+=(--moe-token-dispatcher-type flex --moe-enable-deepep --moe-token-drop-policy probs)
  TRAINING_ARGS+=(--enable-experimental)
  OPT_ARGS+=("deepep_ace")
fi
# 2. CrossEntropy TE fusion.
if [[ "${ENABLE_CE_TE:-0}" -eq 1 ]]; then
  TRAINING_ARGS+=(--cross-entropy-loss-fusion --cross-entropy-fusion-impl te)
  OPT_ARGS+=("ce_te")
fi
# 3. Python manual GC, interval 100.
if [[ "${ENABLE_MANUAL_GC:-0}" -eq 1 ]]; then
  TRAINING_ARGS+=(--manual-gc --manual-gc-interval "${MANUAL_GC_INTERVAL:-100}")
  OPT_ARGS+=("manual_gc")
fi
# 4. RoPE fusion: script forces --no-rope-fusion; rebuild array without it to enable.
if [[ "${ENABLE_ROPE_FUSION:-0}" -eq 1 ]]; then
  _new_model_args=()
  for _a in "${MODEL_ARGS[@]}"; do
    [[ "${_a}" == "--no-rope-fusion" ]] && continue
    _new_model_args+=("${_a}")
  done
  MODEL_ARGS=("${_new_model_args[@]}")
  OPT_ARGS+=("rope_fusion")
fi
# 5. MoE router fusion.
if [[ "${ENABLE_MOE_ROUTER_FUSION:-0}" -eq 1 ]]; then
  MOE_ARGS+=(--moe-router-fusion)
  OPT_ARGS+=("moe_router_fusion")
fi
# 5b. TE RMSNorm → torch fused ATen (gated; musa_patch/rms_norm_fusion.py).
if [[ "${ENABLE_RMSNORM_FUSION:-0}" -eq 1 ]]; then
  OPT_ARGS+=("rmsnorm_fusion")
fi
# 6. gradient-accumulation-fusion: already on (Megatron default True). Optional
#    ENABLE_GRAD_ACCUM_FUSION=0 would add --no-gradient-accumulation-fusion for A/B only.
if [[ "${ENABLE_GRAD_ACCUM_FUSION:-1}" -eq 0 ]]; then
  TRAINING_ARGS+=(--no-gradient-accumulation-fusion)
  OPT_ARGS+=("no_grad_accum_fusion")
fi
# 7. DeepEP-related env from DS internal doc #10/#14/#16 (no ACE).
#    MCCL channels/buffsize; MATE_DEFER_DEEPEP_COUNTS; MUSA_COMPACT_PERMUTE.
if [[ "${ENABLE_DEEPEP_ENV:-0}" -eq 1 ]]; then
  export MCCL_MIN_NCHANNELS="${MCCL_MIN_NCHANNELS:-16}"
  export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-16}"
  export MCCL_BUFFSIZE="${MCCL_BUFFSIZE:-16777216}"
  export MATE_DEFER_DEEPEP_COUNTS="${MATE_DEFER_DEEPEP_COUNTS:-1}"
  export MUSA_COMPACT_PERMUTE="${MUSA_COMPACT_PERMUTE:-1}"
  OPT_ARGS+=("deepep_env")
fi
# 8. shared-expert-overlap (DS #8). Megatron vanilla requires alltoall; flex/DeepEP
#    will reject at config. Kept as a toggle so the failure is explicit.
if [[ "${ENABLE_SHARED_EXPERT_OVERLAP:-0}" -eq 1 ]]; then
  MOE_ARGS+=(--moe-shared-expert-overlap)
  OPT_ARGS+=("shared_expert_overlap")
fi

echo "Experimental opt toggles enabled: ${OPT_ARGS[*]:-(none)}"
echo "USE_DEEPEP_ACE=${USE_DEEPEP_ACE:-0} MCCL_CROSS_NIC=${MCCL_CROSS_NIC:-unset}"
echo "[deepep_ace][input] env TOKEN_NUM=${DEEPEP_ACE_TOKEN_NUM:-unset} HIDDEN=${DEEPEP_ACE_HIDDEN:-unset} NUM_TOPK=${DEEPEP_ACE_NUM_TOPK:-unset} NUM_BUFFERS=${DEEPEP_ACE_NUM_BUFFERS:-unset}"
echo "DeepEP env: MCCL_MIN_NCHANNELS=${MCCL_MIN_NCHANNELS:-unset} MCCL_MAX_NCHANNELS=${MCCL_MAX_NCHANNELS:-unset} MCCL_BUFFSIZE=${MCCL_BUFFSIZE:-unset} MATE_DEFER_DEEPEP_COUNTS=${MATE_DEFER_DEEPEP_COUNTS:-unset} MUSA_COMPACT_PERMUTE=${MUSA_COMPACT_PERMUTE:-unset}"
echo "DeepEP runtime: MUSA_DEVICE_PAGE_SIZE=${MUSA_DEVICE_PAGE_SIZE:-unset} NVSHMEM_IB_SL=${NVSHMEM_IB_SL:-unset} NVSHMEM_IB_TRAFFIC_CLASS=${NVSHMEM_IB_TRAFFIC_CLASS:-unset}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "MiniCPM5 16A3B S5000 node=${NODE_ADDR} rank=${NODE_RANK} TP=${TP_SIZE} PP=${PP_SIZE} CP=${CP_SIZE} EP=${EP_SIZE}"
echo "Torchrun rank logs: ${TORCHRUN_LOG_DIR}/attempt_*/<local_rank>/{stdout,stderr}.log"
echo "Using image Megatron=${MEGATRON_PATH} and MUSA patch=${PATCH_HOME}"

cmd=(
  "${TORCHRUN_BIN[@]}" "${DISTRIBUTED_ARGS[@]}"
  "${PRETRAIN_FILE}"
  "${MODEL_ARGS[@]}"
  "${MOE_ARGS[@]}"
  "${TRAINING_ARGS[@]}"
  "${REGULARIZATION_ARGS[@]}"
  "${LEARNING_RATE_ARGS[@]}"
  "${MODEL_PARALLEL_ARGS[@]}"
  "${MIXED_PRECISION_ARGS[@]}"
  "${DATA_ARGS[@]}"
  "${LOGGING_ARGS[@]}"
  "${TRANSFORMER_ENGINE_ARGS[@]}"
  "${LOAD_ARGS[@]}"
)
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" -eq 1 ]]; then
  echo "DRY_RUN=1; command not executed."
  exit 0
fi
"${cmd[@]}"
