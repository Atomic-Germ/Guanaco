#!/bin/bash
# Does LOAD_MODE decide it?
#
# The first isolation was invalid: it inherited a shell full of LLAMA_ARG_*,
# so every case ran with the working env and all six "survived" - including the
# case meant to be the crashing default.
#
# The real difference between the crashing run and the surviving ones:
#   crashing  LOAD_MODE unset -> AUTO -> plain mmap, KV cache on the GPU
#   surviving LOAD_MODE=streaming -> weights and KV in CPU memory
#
# So the variable is not a KV-layout flag. It is where the KV cache lives.
set -u
B=$(dirname "$(readlink -f "$0")")
BIN=/home/atomic-germ/Projects/Guanaco-Test/build/bin
RUN="$B/run"
MODEL=$(cat "$HOME/guanaco-bench/model_9b.txt")
PORT=8081

try () {
  local label="$1"; shift
  pkill -x llama-server 2>/dev/null
  sleep 3
  # clean slate: no inherited LLAMA_ARG_* at all
  setsid env -i PATH=/usr/bin:/bin HOME="$HOME" \
    LLAMA_ARG_GPU_PILL=1 "$@" \
    "$BIN/llama-server" -m "$MODEL" --port $PORT -ngl 99 -c 4096 -t 6 --no-warmup \
    > "$RUN/mode_$PORT.log" 2>&1 < /dev/null &
  local up=0
  for i in $(seq 1 40); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] \
      && { up=1; break; }
    sleep 3
  done
  if [ $up -eq 0 ]; then echo "  $label: never healthy"; return; fi
  local n=0
  for r in 1 2 3 4 5; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 300 \
      -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d '{"prompt":"Name one invariant of Raft.","n_predict":40,"temperature":0.2}')
    [ "$code" != "200" ] && break
    n=$r
  done
  if pgrep -x llama-server >/dev/null; then
    echo "  $label: SURVIVED $n requests"
  else
    echo "  $label: CRASHED on request $((n+1))"
  fi
}

echo "=== does LOAD_MODE decide it? (9B dense, gpu-pill, -ngl 99, clean env) ==="
try "LOAD_MODE unset (AUTO -> mmap)"   FOO=1
try "LOAD_MODE=mmap"                   LLAMA_ARG_LOAD_MODE=mmap
try "LOAD_MODE=streaming"              LLAMA_ARG_LOAD_MODE=streaming
try "LOAD_MODE=streaming +KV_UNIFIED=0" LLAMA_ARG_LOAD_MODE=streaming LLAMA_ARG_KV_UNIFIED=0
pkill -x llama-server 2>/dev/null
echo "MODE ISOLATION DONE"
