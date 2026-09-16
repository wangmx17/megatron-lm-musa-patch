#!/usr/bin/env bash
set -Eeuo pipefail

# Edit this value when a different training length is needed.
TRAIN_ITERS=100

ROOT=/mbzz_ssd/modelbest_v0.19
PATCH_HOME=${ROOT}/megatron-lm-musa-patch
SCRIPT=${PATCH_HOME}/examples/minicpm5/run_16a3b_2node_v019.sh
INNER_TRAIN=${PATCH_HOME}/examples/minicpm5/train_16a3b_s5000_musa.sh
MEGATRON_PATH=${ROOT}/Megatron-LM
DATA_PATH=${ROOT}/data/minicpm5_real_part00071_text_document
TOKENIZER_MODEL=${ROOT}/models/ckpt
EMERGING_OPTIMIZERS=${ROOT}/emerging-optimizers
RUNS_ROOT=${ROOT}/output_log/v019_2node

HOSTS=(10.124.31.12 10.124.31.10)
SSH_KEY=/mbzz_ssd/zman_mianbi_train/.ssh/id_ed25519_musa_multinode
SSH_PORT=62218
MASTER_PORT=30629
TIMEOUT_SECONDS=21600

SSH=(ssh -i "${SSH_KEY}" -o BatchMode=yes -o IdentitiesOnly=yes
  -o ConnectTimeout=15 -p "${SSH_PORT}")

require_paths() {
  local path
  for path in "${PATCH_HOME}" "${INNER_TRAIN}" "${MEGATRON_PATH}" \
              "${DATA_PATH}.bin" "${DATA_PATH}.idx" "${TOKENIZER_MODEL}" \
              "${EMERGING_OPTIMIZERS}" "${SSH_KEY}"; do
    [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 2; }
  done
}

run_training_on_this_node() {
  local attempt=$1
  local run_root=${RUNS_ROOT}/${attempt}
  local hostfile=${run_root}/hostfile

  export MEGATRON_PATH PATCH_HOME
  export PYTHONPATH=${EMERGING_OPTIMIZERS}
  export PRETRAIN_FILE=${PATCH_HOME}/examples/minicpm5/pretrain_minicpm5_musa.py
  export EP_SIZE=8 CP_SIZE=8 EXPERT_TP_SIZE=1 GPUS_PER_NODE=8
  export TRAIN_ITERS LR_WARMUP_ITERS=1000 SAVE_INTERVAL=999999
  export INITIAL_LOSS_SCALE=4294967296
  export SEQ_LENGTH=131072 MAX_POSITION_EMBEDDINGS=131072 SEED=1234
  export ALLOW_RANDOM_INIT=1 RESUME=0 NO_SAVE=1
  export SAVE_DIR=${run_root}/checkpoint TENSORBOARD_DIR=${run_root}/tensorboard
  export OUTPUT_LOG_ROOT=${run_root}/rank_logs
  export LOG_TEE_LEVEL=3 DISABLE_RECOMPUTE=1
  export ENABLE_DEEPEP=0 USE_DEEPEP_ACE=0 DEEPEP_ACE_DYNAMIC_TOKEN_NUM=1
  export ENABLE_CE_TE=1 ENABLE_MANUAL_GC=1 ENABLE_ROPE_FUSION=1
  export ENABLE_MOE_ROUTER_FUSION=1 ENABLE_GRAD_ACCUM_FUSION=1 ENABLE_DEEPEP_ENV=1
  export ENABLE_RMSNORM_FUSION=0 USE_GROUPED_GEMM=1 MOE_DEEPEP_NUM_SMS=56
  export USE_SPAN_BASED_ATTN=1 USE_FLASH_ATTN=1 OPTIMIZER=muon
  export MUON_BATCH_NS=0 MUON_DP1_LOW_MEMORY=0 MUON_FUSED_POINTWISE=0
  export MUSA_TE_PADDED_METADATA_NO_SYNC=0 MUSA_FUSED_ROUTE_CONVERSION=0
  export MUON_TE_EXPERT_BATCH_NS=0 ENABLE_FLEX_SHARED_EXPERT_EAGER=0
  export ENABLE_SHARED_EXPERT_OVERLAP=0 NVTE_BATCH_MHA_P2P_COMM=0 ENABLE_PROFILER=0
  export MCCL_MIN_NCHANNELS=16 MCCL_MAX_NCHANNELS=16 MCCL_BUFFSIZE=16777216
  export PYTHONDONTWRITEBYTECODE=1 MEGATRON_LOG_RUNTIME_PATHS=1

  bash "${INNER_TRAIN}" "${run_root}" "${PATCH_HOME}" \
    minicpm5_v019_tp2_pp1_cp8_ep8_dp1_seq131072_gbs32 \
    "${hostfile}" "${DATA_PATH}" 2 1 1 32 "${TOKENIZER_MODEL}" \
    "v019_2node_${attempt}" "${MASTER_PORT}"
}

supervise_node() {
  local attempt=$1 host=$2
  local run_root=${RUNS_ROOT}/${attempt}
  local pidfile=${run_root}/node.${host}.pid
  local job sampler rc=0

  setsid timeout --signal=TERM --kill-after=30s "${TIMEOUT_SECONDS}" \
    bash "${SCRIPT}" __train "${attempt}" &
  job=$!
  printf '%s\n' "${job}" > "${pidfile}"
  (
    while kill -0 "${job}" 2>/dev/null; do
      date -Is
      mthreads-gmi | awk '/^[0-7][[:space:]]+MTT S5000/'
      sleep 5
    done
  ) > "${run_root}/memory.${host}.log" &
  sampler=$!
  wait "${job}" || rc=$?
  wait "${sampler}" || true
  printf '%s\n' "${rc}" > "${run_root}/node.${host}.rc"
  date -Is > "${run_root}/node.${host}.finished_at"
  mthreads-gmi > "${run_root}/gpu_final.${host}.log"
  return "${rc}"
}

stop_node() {
  local attempt=$1 host=$2
  local pidfile=${RUNS_ROOT}/${attempt}/node.${host}.pid
  local job pgid cmdline
  [[ -f "${pidfile}" ]] || return 0
  read -r job < "${pidfile}"
  [[ "${job}" =~ ^[0-9]+$ && -r /proc/${job}/cmdline ]] || return 0
  cmdline=$(tr '\0' ' ' < "/proc/${job}/cmdline")
  [[ "${cmdline}" == *"${SCRIPT}"* && "${cmdline}" == *"__train ${attempt}"* ]] || return 0
  pgid=$(ps -o pgid= -p "${job}" | tr -d ' ')
  [[ "${pgid}" == "${job}" ]] && kill -TERM -- "-${job}"
}

case "${1:-}" in
  __train)
    require_paths
    run_training_on_this_node "$2"
    exit $?
    ;;
  __node)
    supervise_node "$2" "$3"
    exit $?
    ;;
  __stop)
    stop_node "$2" "$3"
    exit $?
    ;;
esac

require_paths
attempt=${V019_ATTEMPT:-$(date +%Y%m%d_%H%M%S)}
run_root=${RUNS_ROOT}/${attempt}
mkdir -p "${run_root}"
printf '%s\n' \
  '10.124.31.12 slots=8' \
  '10.124.31.10 slots=8' > "${run_root}/hostfile"
printf '%s\n' "${run_root}" > "${RUNS_ROOT}/latest_run.txt"
date -Is > "${run_root}/started_at"
printf '%s\n' "${BASHPID}" > "${run_root}/launcher.pid"

for host in "${HOSTS[@]}"; do
  "${SSH[@]}" root@"${host}" 'mthreads-gmi; ss -ltn' \
    > "${run_root}/preflight.${host}.log"
  grep -q 'No running processes found' "${run_root}/preflight.${host}.log" || {
    echo "GPU process found on ${host}; training not started." >&2
    exit 8
  }
  grep -q ":${MASTER_PORT} " "${run_root}/preflight.${host}.log" && {
    echo "Port ${MASTER_PORT} is busy on ${host}; training not started." >&2
    exit 9
  }
done

cleanup() {
  local final_rc=$?
  trap - EXIT
  for host in "${HOSTS[@]}"; do
    "${SSH[@]}" root@"${host}" bash "${SCRIPT}" __stop "${attempt}" "${host}" || true
  done
  printf '%s\n' "${final_rc}" > "${run_root}/launcher.rc"
  date -Is > "${run_root}/finished_at"
  exit "${final_rc}"
}
trap cleanup EXIT

for host in "${HOSTS[@]}"; do
  "${SSH[@]}" root@"${host}" bash "${SCRIPT}" __node "${attempt}" "${host}" \
    > "${run_root}/node.${host}.log" 2>&1 &
done

remaining=${#HOSTS[@]}
while (( remaining > 0 )); do
  if wait -n; then
    remaining=$((remaining - 1))
  else
    rc=$?
    echo "A node failed with rc=${rc}; stopping this attempt." >&2
    exit "${rc}"
  fi
done
