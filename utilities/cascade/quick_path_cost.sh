#!/bin/bash
# What does streaming cost a model that does not need paging?
#
# Established: the KV state-save crash is a plain-llama.cpp bug, triggered by a
# Vulkan-backed KV cache under llama-server. LOAD_MODE=streaming is immune
# because it keeps the KV in host memory.
#
# That makes streaming the fast AND stable route to a GPU-resident model - but
# only if it does not cost anything on a model that fits in RAM. The quick model
# of the cascade fits, so this measures the tax.
#
#   ngl 0, mmap                 CPU compute, no GPU        (the old workaround)
#   ngl 99, mmap                GPU compute, KV on GPU     -> CRASHES
#   ngl 99, streaming           GPU compute, KV in host    -> should be stable
#   ngl 99, streaming, tiny budget  forces paging anyway   -> the stress case
set -u
B=$(dirname "$(readlink -f "$0")")
BIN=/home/atomic-germ/Projects/Guanaco-Test/build/bin
RUN="$B/run"
MODEL=$(cat "$HOME/guanaco-bench/model_9b.txt")
PORT=8081

try () {
  local label="$1"; shift
  # split env assignments from CLI flags: `env` treats a leading -ngl as its
  # own option, which silently broke the first version of this script.
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
    > "$RUN/q_$PORT.log" 2>&1 < /dev/null &
  local up=0
  for i in $(seq 1 60); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] \
      && { up=1; break; }
    sleep 3
  done
  if [ $up -eq 0 ]; then echo "  $label: never healthy"; return; fi
  local rates=() n=0
  for r in 1 2 3 4 5; do
    out=$(curl -s --max-time 600 -X POST "http://127.0.0.1:$PORT/completion" \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"Name one invariant of Raft and explain it.","n_predict":120,"temperature":0.2}' \
      | python3 -c "import json,sys;print('%.2f'%json.load(sys.stdin)['timings']['predicted_per_second'])" 2>/dev/null)
    [ -z "$out" ] && break
    rates+=("$out"); n=$r
  done
  if pgrep -x llama-server >/dev/null; then
    echo "  $label: survived $n req, rates: ${rates[*]}"
  else
    echo "  $label: CRASHED on request $((n+1))"
  fi
}

echo "=== 9B dense: what does each route cost? ==="
try "ngl 0,  mmap (CPU compute)"           -ngl 0
try "ngl 99, mmap (GPU, KV on GPU)"        LLAMA_ARG_GPU_PILL=1 -ngl 99
try "ngl 99, streaming (GPU, KV in host)"  LLAMA_ARG_GPU_PILL=1 LLAMA_ARG_LOAD_MODE=streaming -ngl 99
try "ngl 99, streaming, 2GB budget"        LLAMA_ARG_GPU_PILL=1 LLAMA_ARG_LOAD_MODE=streaming \
                                            LLAMA_ARG_GUANACO_DENSE_BUDGET=2048 -ngl 99
pkill -x llama-server 2>/dev/null
echo "COST DONE"
