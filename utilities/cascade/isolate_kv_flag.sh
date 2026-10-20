#!/bin/bash
# Which flag makes the KV state-save crash go away?
#
# The crash persists after the Lunar SDK rebuild, with an identical stack:
#   get_available_slot -> prompt_save -> state_seq_get_data
#     -> llama_io_write_host -> ggml_backend_vk_buffer_get_tensor
#     -> ggml_vk_buffer_read_2d   <- SIGSEGV
#
# The 111B run that worked all day used env the default does not:
#   LLAMA_ARG_KV_UNIFIED=0   LLAMA_ARG_CACHE_TYPE_K/V=q8_0   LLAMA_ARG_FLASH_ATTN=1
# The KV layout decides what state_seq_get_data copies and how big the
# destination is, so one of those is the likely discriminator.
#
# Varies ONE flag at a time from the crashing default.
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
  setsid env "$@" LLAMA_ARG_GPU_PILL=1 \
    "$BIN/llama-server" -m "$MODEL" --port $PORT -ngl 99 -c 4096 -t 6 --no-warmup \
    > "$RUN/flag_$PORT.log" 2>&1 < /dev/null &
  local up=0
  for i in $(seq 1 40); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] \
      && { up=1; break; }
    sleep 3
  done
  if [ $up -eq 0 ]; then echo "  $label: never healthy"; return; fi
  local crashed=0
  for r in 1 2 3 4 5; do
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 300 \
      -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
      -d '{"prompt":"Name one invariant of Raft.","n_predict":40,"temperature":0.2}')
    [ "$code" != "200" ] && { crashed=1; break; }
  done
  if pgrep -x llama-server >/dev/null; then
    echo "  $label: SURVIVED 5 requests"
  else
    echo "  $label: CRASHED on request $((crashed+1))"
  fi
}

echo "=== isolating the flag (9B dense, gpu-pill, -ngl 99) ==="
try "default (known crash)"
try "KV_UNIFIED=0"            LLAMA_ARG_KV_UNIFIED=0
try "FLASH_ATTN=1"            LLAMA_ARG_FLASH_ATTN=1
try "CACHE q8_0/q8_0"         LLAMA_ARG_CACHE_TYPE_K=q8_0 LLAMA_ARG_CACHE_TYPE_V=q8_0
try "KV_UNIFIED=0 + FLASH_ATTN=1" LLAMA_ARG_KV_UNIFIED=0 LLAMA_ARG_FLASH_ATTN=1
try "full working env"        LLAMA_ARG_KV_UNIFIED=0 LLAMA_ARG_FLASH_ATTN=1 \
                              LLAMA_ARG_CACHE_TYPE_K=q8_0 LLAMA_ARG_CACHE_TYPE_V=q8_0
pkill -x llama-server 2>/dev/null
echo "FLAG ISOLATION DONE"
