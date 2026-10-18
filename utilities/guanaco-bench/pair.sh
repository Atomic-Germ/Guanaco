#!/bin/bash
# Bring up the cascade pair: a small model always resident, and a large model
# under Guanaco streaming, behind two llama-server instances.
#
# Functionality only. Which requests go where is the efficacy question, and that
# needs a labelled workload rather than a synthetic one.
#
# Usage: pair.sh start [dense|moe] | stop | status
set -u
B=$(dirname "$(readlink -f "$0")")
BIN=/home/atomic-germ/Projects/Guanaco-Test/build/bin
SMALL=$(cat "$B/model_8b.txt")
RUN=/home/atomic-germ/guanaco-bench/run
mkdir -p "$RUN"

large_for () {
  case "$1" in
    dense) cat "$B/model_path.txt" ;;   # Qwen3.8-27B, 22 GB dense
    moe)   cat "$B/model_moe.txt" ;;    # Huihui-MoE-24B-A8B, 20 GB MoE
    *) echo "unknown large model: $1" >&2; exit 2 ;;
  esac
}

stop_all () {
  pkill -f "llama-server.*--port 808" 2>/dev/null
  pkill -f "llama-server.*--port 809" 2>/dev/null
  sleep 2
  echo "stopped"
}

case "${1:-status}" in
  start)
    KIND="${2:-dense}"
    BIG=$(large_for "$KIND")
    [ -s "$BIG" ] || { echo "bad large model path" >&2; exit 1; }
    stop_all >/dev/null

    # Small model: it fits, so plain mmap and no streaming. Auto-load-mode would
    # arrive at the same place, but being explicit here keeps the intent clear.
    # setsid: these must outlive this script. Without it they are children of a
    # shell that exits as soon as both are healthy, and they were observed
    # vanishing mid-request with no error in their logs.
    # Small model: it fits, so plain mmap and no streaming.
    #
    # -ngl 0, not 99. llama-server + any Vulkan-backed KV cache segfaults on the
    # second request in ggml_vk_buffer_read_2d, reached from server_slot::
    # prompt_save -> state_seq_get_data -> llama_io_write_host. Reproduced 4/4 with
    # an identical stack, with gpu-pill on AND off, so it is not the UMA
    # allocation; it is upstream Vulkan, triggered by the KV cache being in a
    # Vulkan buffer at all. Costs ~6% throughput versus gpu-pill's 7.6 tok/s,
    # which is worth paying against a deterministic crash.
    echo "starting small (AesCode-8B, 5.2 GB) on :8081"
    setsid env -u LLAMA_ARG_LOAD_MODE -u LLAMA_ARG_GPU_PILL \
      "$BIN/llama-server" -m "$SMALL" --port 8081 -ngl 0 -c 4096 -t 6 \
      --no-warmup > "$RUN/small.log" 2>&1 < /dev/null &
    echo $! > "$RUN/small.pid"

    # Large model: streamed and paged, with its own budget.
    echo "starting large ($KIND, $(echo "scale=1; $(stat -Lc%s "$BIG")/1000000000" | bc) GB) on :8082"
    setsid env LLAMA_ARG_LOAD_MODE=streaming LLAMA_ARG_GPU_PILL=1 LLAMA_ARG_KV_UNIFIED=0 \
        LLAMA_ARG_CACHE_TYPE_K=q8_0 LLAMA_ARG_CACHE_TYPE_V=q4_0 LLAMA_ARG_THREADS=4 \
        LLAMA_ARG_GUANACO_DENSE_BUDGET=${LARGE_BUDGET:-7000} \
      "$BIN/llama-server" -m "$BIG" --port 8082 -ngl 99 -c 2048 -t 4 \
      --no-warmup > "$RUN/large.log" 2>&1 < /dev/null &
    echo $! > "$RUN/large.pid"

    echo "small  pid $(cat "$RUN/small.pid")  log $RUN/small.log"
    echo "large  pid $(cat "$RUN/large.pid")  log $RUN/large.log"
    echo
    echo "waiting for both to come up (large loads from disk, this takes a while)..."
    for i in $(seq 1 60); do
      s=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8081/health 2>/dev/null)
      l=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8082/health 2>/dev/null)
      [ "$s" = "200" ] && [ "$l" = "200" ] && { echo "both ready after ~$((i*10))s"; exit 0; }
      sleep 10
    done
    echo "timed out; check the logs" >&2
    exit 1
    ;;
  stop)   stop_all ;;
  status)
    for p in 8081 8082; do
      printf ":%s  %s\n" "$p" "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$p/health 2>/dev/null || echo down)"
    done
    free -g | head -2
    ;;
  *) echo "usage: $0 start [dense|moe] | stop | status" >&2; exit 2 ;;
esac
