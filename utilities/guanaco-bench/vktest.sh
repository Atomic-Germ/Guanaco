#!/bin/bash
# Does --no-cache-prompt avoid the Vulkan segfault?
#
# Confirmed so far: llama-server with -ngl 99 + gpu-pill segfaults on the second
# request inside ggml_vk_buffer_read_2d, reached from server_slot::prompt_save
# -> llama_context::state_seq_get_data -> llama_io_write_host. With -ngl 0 the
# same workload survives. So it is Vulkan-specific, in the slot-state save path.
#
# prompt_save only runs when the prompt cache is enabled, so disabling it should
# avoid the read entirely - at the cost of losing slot reuse.
set -u
B=$(dirname "$(readlink -f "$0")")
BIN=/home/atomic-germ/Projects/Guanaco-Test/build/bin
SMALL=$(cat "$B/model_8b.txt")

# exact-name match: a -f pattern would also match the shell running this script
pkill -x llama-server 2>/dev/null
sleep 3

run_case () {
  local label="$1"; shift
  echo "=== $label ==="
  setsid env -u LLAMA_ARG_LOAD_MODE -u LLAMA_ARG_GPU_PILL \
    "$BIN/llama-server" -m "$SMALL" --port 8081 -c 4096 -t 6 --no-warmup "$@" \
    > "$B/run/case.log" 2>&1 < /dev/null &
  for i in $(seq 1 24); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8081/health 2>/dev/null)" = "200" ] && break
    sleep 5
  done
  echo "  health: $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8081/health)"
  for r in 1 2 3 4; do
    printf "  req %d: " $r
    python3 - << 'PY'
import json, urllib.request, sys
body = json.dumps({"prompt": "Name one invariant of Raft.",
                   "n_predict": 40, "temperature": 0.2, "stream": False}).encode()
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8081/completion", data=body,
        headers={"Content-Type": "application/json"}), timeout=300))
    print("%.2f tok/s" % d["timings"]["predicted_per_second"])
except Exception as e:
    print("FAILED:", e); sys.exit(1)
PY
    [ $? -ne 0 ] && break
  done
  if pgrep -x llama-server >/dev/null; then echo "  server: alive"; else echo "  server: DEAD"; fi
  pkill -x llama-server 2>/dev/null
  sleep 3
}

run_case "gpu-pill, prompt cache ON (known crash)" -ngl 99
run_case "gpu-pill, --no-cache-prompt"          -ngl 99 --no-cache-prompt
echo "VULKAN CASES DONE"
