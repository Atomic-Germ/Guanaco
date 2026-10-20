#!/bin/bash
# Diagnostic: what does the EE specialist actually return?
#
# The specialist scored 0/8 on its own domain while answering 6x faster than the
# generalist, which is the signature of a server that is not doing the work
# rather than a model that is weak. Print one raw response and the server log.
set -u
B=$(dirname "$(readlink -f "$0")")
BIN=/home/atomic-germ/Projects/Guanaco-Test/build/bin
SPEC=$(cat "$HOME/guanaco-bench/model_ee.txt")

pkill -x llama-server 2>/dev/null
sleep 2

echo "loading specialist alone: $SPEC"
ls -laL "$SPEC" | awk '{printf "  size %.2f GB\n", $5/1e9}'
setsid env -u LLAMA_ARG_LOAD_MODE -u LLAMA_ARG_GPU_PILL \
  "$BIN/llama-server" -m "$SPEC" --port 8082 -ngl 0 -c 4096 -t 6 --no-warmup \
  > /tmp/spec_diag.log 2>&1 < /dev/null &

for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8082/health)" = "200" ] \
    && { echo "ready after ~$((i*5))s"; break; }
  sleep 5
done
echo "health: $(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8082/health)"

echo
echo "=== raw response to an EE question ==="
curl -s --max-time 300 -X POST http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"A 12 V source drives a 470 ohm resistor. What current flows, in milliamps? Number only."}],"max_tokens":500,"temperature":0.0}' \
  | head -c 900
echo
echo
echo "=== also via /completion (raw) ==="
curl -s --max-time 300 -X POST http://127.0.0.1:8082/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"A 12 V source drives a 470 ohm resistor. What current flows, in milliamps? Number only.","n_predict":120,"temperature":0.0}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('content:', repr(d.get('content',''))[:400]); print('tokens:', d.get('tokens_predicted'))"

echo
echo "=== server log tail ==="
tail -12 /tmp/spec_diag.log
