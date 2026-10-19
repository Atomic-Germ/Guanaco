#!/bin/bash
# Is the Vulkan segfault ours, or upstream?
#
# The crash is in ggml_vk_buffer_read_2d (upstream code; our Vulkan patch adds
# no memmove/memcpy) reached from llama-server's slot-state save. Both -ngl 99
# with gpu-pill and -ngl 99 without it put the KV cache in a Vulkan buffer, but
# only gpu-pill allocates it through our UMA buffer type.
#
#   gpu-pill ON   -> KV cache in a UMA buffer (our allocation path)
#   gpu-pill OFF  -> KV cache in a normal Vulkan buffer (upstream allocation)
#
# Both crashing  => upstream Vulkan bug, our patch merely makes it reachable
# only-on crashing => our UMA allocation is implicated
# neither crashing => something else
set -u
B=$(dirname "$(readlink -f "$0")")
BIN=/home/atomic-germ/Projects/Guanaco-Test/build/bin
SMALL=$(cat "$B/model_8b.txt")

pkill -x llama-server 2>/dev/null
sleep 3

case_ () {
  local label="$1" pill="$2" ngl="$3"
  echo "=== $label (gpu_pill=$pill, ngl=$ngl) ==="
  if [ "$pill" = "auto" ]; then
    setsid env -u LLAMA_ARG_GPU_PILL "$BIN/llama-server" -m "$SMALL" --port 8081 \
      -ngl "$ngl" -c 4096 -t 6 --no-warmup > "$B/run/case.log" 2>&1 < /dev/null &
  else
    setsid env LLAMA_ARG_GPU_PILL="$pill" "$BIN/llama-server" -m "$SMALL" --port 8081 \
      -ngl "$ngl" -c 4096 -t 6 --no-warmup > "$B/run/case.log" 2>&1 < /dev/null &
  fi
  local up=0
  for i in $(seq 1 30); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8081/health 2>/dev/null)" = "200" ] && { up=1; break; }
    sleep 5
  done
  [ $up -eq 0 ] && { echo "  never became healthy; log:"; tail -3 "$B/run/case.log"; return; }
  for r in 1 2 3; do
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
  if pgrep -x llama-server >/dev/null; then echo "  server: alive"; else echo "  server: DEAD (crashed)"; fi
  pkill -x llama-server 2>/dev/null
  sleep 4
}

case_ "gpu-pill OFF, full offload" 0 99
case_ "gpu-pill OFF, no offload"    0 0
case_ "gpu-pill ON (UMA)"           1 99
echo "ATTRIBUTION DONE"
