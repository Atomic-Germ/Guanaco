#!/bin/bash
# Is the trigger the KV cache living in a Vulkan buffer?
#
# The evidence points that way:
#   - streaming forces n_gpu_layers=0, so the weights are in host memory, yet
#     the Shirley-8B still died partway into the second request, on the
#     "selected slot by LCP similarity" path, which shifts the cached KV
#   - the KV cache location is governed by offload_kqv, not by n_gpu_layers
#   - with gpu_pill it works fine, and gpu_pill keeps things in host memory
#
# If so, --no-kv-offload is a one-flag mitigation, and the crash is a plain
# llama.cpp bug in the Vulkan-backed KV path rather than anything to do with
# the UMA buffer type.
set -u
B=$(dirname "$(readlink -f "$0")")
BIN=/home/atomic-germ/Projects/Guanaco-Test/build/bin
RUN="$B/run"
MODEL=$(cat "$HOME/guanaco-bench/model_9b.txt")
PORT=8081

try () {
  local label="$1"; shift
  local envs=() flags=() phase=env
  for a in "$@"; do
    if [ "$phase" = env ]; then
      case "$a" in
        -*) phase=flags; flags+=("$a") ;;
        *) envs+=("$a") ;;
      esac
    else
      flags+=("$a")
    fi
  done
  pkill -x llama-server 2>/dev/null
  sleep 3
  setsid env -i PATH=/usr/bin:/bin HOME="$HOME" "${envs[@]}" \
    "$BIN/llama-server" -m "$MODEL" --port $PORT -c 4096 -t 6 --no-warmup "${flags[@]}" \
    > "$RUN/kvo_$PORT.log" 2>&1 < /dev/null &
  local up=0
  for i in $(seq 1 60); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] \
      && { up=1; break; }
    sleep 3
  done
  if [ $up -eq 0 ]; then echo "  $label: never healthy"; return; fi
  local n=0
  for r in 1 2 3 4 5; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 600 \
      -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d '{"prompt":"Name one invariant of Raft and explain it.","n_predict":120,"temperature":0.2}')
    [ "$code" != "200" ] && break
    n=$r
  done
  if pgrep -x llama-server >/dev/null; then
    echo "  $label: SURVIVED $n requests"
  else
    echo "  $label: CRASHED on request $((n+1))"
  fi
}

echo "=== is it the KV cache being on the GPU? (9B, gpu-pill, -ngl 99) ==="
try "kv offload ON (default)"  LLAMA_ARG_GPU_PILL=1 -ngl 99
try "kv offload OFF"          LLAMA_ARG_GPU_PILL=1 -ngl 99 --no-kv-offload
try "kv offload OFF, ngl 0"   -ngl 0 --no-kv-offload
pkill -x llama-server 2>/dev/null
echo "KVO DONE"
