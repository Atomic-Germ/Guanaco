#!/bin/bash
# Phase 1: what is not broken any longer?
#
# The Vulkan crash was: llama-server with -ngl 99 + gpu_pill segfaulted on the
# SECOND request, in ggml_vk_buffer_read_2d, reached from server_slot::
# prompt_save -> llama_context::state_seq_get_data -> llama_io_write_host.
# Reproduced 4/4 with identical stacks, gpu-pill on and off, so it was not the
# UMA allocation - it was the Vulkan stack. A locally built Lunar SDK has since
# replaced it, and this checks whether that actually fixed the path.
#
# Both architectures are exercised because the dense and MoE paths reach
# different buffers.
#
# Uses the 8B (dense) and the 24B-A8B (MoE) - known-good files, so a failure is
# unambiguously ours.
set -u
B=$(dirname "$(readlink -f "$0")")
BIN=/home/atomic-germ/Projects/Guanaco-Test/build/bin
RUN="$B/run"
mkdir -p "$RUN"

MODEL_DENSE=$(cat "$HOME/guanaco-bench/model_9b.txt")
MODEL_MOE=$(cat "$HOME/guanaco-bench/model_moe.txt")

check_model () {
  local label="$1" path="$2" port="$3" ngll="$4"
  echo "### $label ###"
  pkill -x llama-server 2>/dev/null
  sleep 3
  setsid env -u LLAMA_ARG_LOAD_MODE -u LLAMA_ARG_GPU_PILL LLAMA_ARG_GPU_PILL=1 \
    "$BIN/llama-server" -m "$path" --port "$port" -ngl "$ngll" -c 4096 -t 6 --no-warmup \
    > "$RUN/verify_$port.log" 2>&1 < /dev/null &
  local ok=0
  for i in $(seq 1 60); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$port/health 2>/dev/null)" = "200" ] \
      && { ok=1; echo "  loaded in ~$((i*5))s"; break; }
    sleep 5
  done
  if [ $ok -eq 0 ]; then
    echo "  NEVER HEALTHY:"; tail -5 "$RUN/verify_$port.log" | sed 's/^/    /'
    return 1
  fi
  for r in 1 2 3 4 5 6; do
    printf "  req %d: " $r
    code=$(curl -s -o /tmp/vresp.json -w '%{http_code}' --max-time 900 \
      -X POST "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
      -d '{"prompt":"Name one invariant of Raft and explain it briefly.","n_predict":60,"temperature":0.2}')
    if [ "$code" != "200" ]; then
      echo "HTTP $code  <-- CRASH?"
      break
    fi
    rate=$(python3 -c "import json;d=json.load(open('/tmp/vresp.json'));print('%.2f tok/s'%d['timings']['predicted_per_second'])" 2>/dev/null || echo "?")
    echo "ok  $rate"
  done
  if pgrep -x llama-server >/dev/null; then
    echo "  server: ALIVE after the sequence"
  else
    echo "  server: DEAD  <-- reproduced the crash"
    dmesg -T 2>/dev/null | tail -3 | sed 's/^/    /'
  fi
}

echo "=========== dense (8B), gpu-pill, -ngl 99 ==========="
check_model "9B dense (Qwen3.8-9B-Distill)" "$MODEL_DENSE" 8081 99

echo
echo "=========== MoE (24B-A8B), gpu-pill, -ngl 99 ==========="
check_model "24B MoE" "$MODEL_MOE" 8082 99

pkill -x llama-server 2>/dev/null
echo
echo "VERIFY DONE"
