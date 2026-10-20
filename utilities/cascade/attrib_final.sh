#!/bin/bash
# Attribution, run in sequence so the two servers never contend for the port.
set -u
B=$(dirname "$(readlink -f "$0")")
MODEL=$(cat "$HOME/guanaco-bench/model_9b.txt")
PORT=8081
RUN="$HOME/guanaco-bench/run"
mkdir -p "$RUN"
OUR=/home/atomic-germ/Projects/Guanaco-Test/build/bin/llama-server
VAN=/home/atomic-germ/vanilla/llama-cpp/build/bin/llama-server

case_ () {
  local label="$1" bin="$2"; shift 2
  pkill -x llama-server 2>/dev/null
  sleep 3
  # flags go AFTER the binary: `env` treats a leading -ngl as its own option
  setsid env -i PATH=/usr/bin:/bin HOME="$HOME" \
    "$bin" -m "$MODEL" --port $PORT -c 4096 -t 6 --no-warmup "$@" \
    > "$RUN/attr_$PORT.log" 2>&1 < /dev/null &
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

echo "=== OURS (Guanaco patches) ==="
case_ "ours, -ngl 99, kv on GPU"  "$OUR" -ngl 99
case_ "ours, -ngl 99, --no-kv-offload" "$OUR" -ngl 99 --no-kv-offload

echo
echo "=== VANILLA upstream (no Guanaco patches) ==="
case_ "vanilla, -ngl 99"  "$VAN" -ngl 99
case_ "vanilla, -ngl 99, --no-kv-offload" "$VAN" -ngl 99 --no-kv-offload
case_ "vanilla, -ngl 0"   "$VAN" -ngl 0

pkill -x llama-server 2>/dev/null
echo "ATTRIBUTION DONE"
