#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mbzz_ssd/modelbest_v0.19
export V019_ATTEMPT=${V019_ATTEMPT:-$(date +%Y%m%d_%H%M%S)}
RUN=${ROOT}/v019_compat_50step_20260915/${V019_ATTEMPT}
SSH=(ssh -i /mbzz_ssd/zman_mianbi_train/.ssh/id_ed25519_musa_multinode
  -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=15 -p 62218)
HOSTS=(10.124.31.12 10.124.31.10)
mkdir -p "$RUN"
printf '%s\n' "$RUN" > "${ROOT}/v019_compat_50step_20260915/latest_run.txt"
date -Is > "$RUN/started_at"
printf '%s\n' "$BASHPID" > "$RUN/launcher.pid"
for host in "${HOSTS[@]}"; do
  "${SSH[@]}" root@"$host" 'mthreads-gmi; ss -ltn' > "$RUN/preflight.${host}.log"
  grep -q 'No running processes found' "$RUN/preflight.${host}.log"
  if grep -q ':30629 ' "$RUN/preflight.${host}.log"; then
    echo 'Rendezvous port busy; training not started.' >&2
    exit 9
  fi
done
cleanup() {
  local final_rc=$?
  trap - EXIT
  for host in "${HOSTS[@]}"; do
    "${SSH[@]}" root@"$host" bash "${ROOT}/v019_50step_node_supervisor.sh" stop "$V019_ATTEMPT" "$host" || true
  done
  printf '%s\n' "$final_rc" > "$RUN/launcher.rc"
  date -Is > "$RUN/finished_at"
  exit "$final_rc"
}
trap cleanup EXIT
pids=()
for host in "${HOSTS[@]}"; do
  "${SSH[@]}" root@"$host" bash "${ROOT}/v019_50step_node_supervisor.sh" run "$V019_ATTEMPT" "$host" \
    > "$RUN/node.${host}.log" 2>&1 &
  pids+=("$!")
done
remaining=2
while (( remaining > 0 )); do
  if wait -n; then
    remaining=$((remaining-1))
  else
    rc=$?
    echo "A node failed with rc=$rc; stopping only this attempt's process groups."
    exit "$rc"
  fi
done
