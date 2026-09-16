#!/usr/bin/env bash
set -Eeuo pipefail
mode=$1
export V019_ATTEMPT=$2
host=$3
[[ "$V019_ATTEMPT" =~ ^[0-9]{8}_[0-9]{6}$ ]] || exit 2
[[ "$host" == 10.124.31.12 || "$host" == 10.124.31.10 ]] || exit 2
ROOT=/mbzz_ssd/modelbest_v0.19
RUN=${ROOT}/v019_compat_100step_loss_scale_2p32_20260915/${V019_ATTEMPT}
PIDFILE=$RUN/node.${host}.pid
if [[ "$mode" == stop ]]; then
  if [[ -f "$PIDFILE" ]]; then
    read -r job < "$PIDFILE"
    if [[ "$job" =~ ^[0-9]+$ && -r /proc/$job/cmdline ]] && \
       tr '\0' ' ' < /proc/$job/cmdline | grep -q '/mbzz_ssd/modelbest_v0.19/run_v019_100step_node.sh' && \
       tr '\0' '\n' < /proc/$job/environ | grep -qx "V019_ATTEMPT=$V019_ATTEMPT"; then
      pgid=$(ps -o pgid= -p "$job" | tr -d ' ')
      [[ "$pgid" == "$job" ]] && kill -TERM -- "-$job"
    fi
  fi
  exit 0
fi
[[ "$mode" == run ]] || exit 2
setsid timeout --signal=TERM --kill-after=30s 21600 \
  bash ${ROOT}/run_v019_100step_node.sh &
job=$!
printf '%s\n' "$job" > "$PIDFILE"
(
  while kill -0 "$job" 2>/dev/null; do
    date -Is
    mthreads-gmi | awk '/^[0-7][[:space:]]+MTT S5000/'
    sleep 5
  done
) > "$RUN/memory.${host}.log" &
sampler=$!
rc=0
wait "$job" || rc=$?
wait "$sampler" || true
printf '%s\n' "$rc" > "$RUN/node.${host}.rc"
date -Is > "$RUN/node.${host}.finished_at"
mthreads-gmi > "$RUN/gpu_final.${host}.log"
exit "$rc"
