#!/bin/bash
# Decisive attribution: does UNPATCHED upstream llama.cpp have the same crash?
#
# Vanilla llama.cpp at the pinned commit aa94f2086, Vulkan enabled, no Guanaco
# patches of any kind. The scenario is the one that crashed in our build:
# a GPU-resident KV cache under llama-server, multiple sequential requests.
#
# If it crashes here too, it is a plain llama.cpp bug in the Vulkan-backed KV
# path, reachable with -ngl 99 on any Vulkan backend, and the reproducer is
# small enough to file upstream. If it survives, one of our patches is
# implicated and the earlier attribution was wrong.
set -u
B=$(dirname "$(readlink -f "$0")")
MODEL=$(cat "$HOME/guanaco-bench/model_9b.txt")
PORT=8081
RUN="$HOME/guanaco-bench/run"
mkdir -p "$RUN"

run_case () {
  local label="$1" bin="$2"; shift 2
  pkill -x llama-server 2>/dev/null
  sleep 3
  setsid env -i PATH=/usr/bin:/bin HOME="$HOME" "$@" \
    "$bin" -m "$MODEL" --port $PORT -c 4096 -t 6 --no-warmup \
    > "$RUN/vanilla_$PORT.log" 2>&1 < /dev/null &
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
    coredumpctl list --no-pager 2>/dev/null | tail -1 | awk '{print "    latest coredump pid " $5}'
  fi
}

echo "=== vanilla upstream llama.cpp (no Guanaco patches) ==="
run_case "vanilla, -ngl 99"  /home/atomic-germ/vanilla/llama-cpp/build/bin/llama-server -ngl 99
run_case "vanilla, -ngl 99, --no-kv-offload" /home/atomic-germ/vanilla/llama-cpp/build/bin/llama-server -ngl 99 --no-kv-offload
run_case "vanilla, -ngl 0"   /home/atomic-germ/vanilla/llama-cpp/build/bin/llama-server -ngl 0
pkill -x llama-server 2>/dev/null
echo "VANILLA ATTRIBUTION DONE"
