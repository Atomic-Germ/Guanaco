#!/bin/bash
# Cold start: the whole basis of the resident / on-demand switch.
#
# "Load time" with the model already in page cache is not a number anyone pays,
# so the cache is dropped first. What a user actually waits is two things:
#
#   to_first_token  after asking for the large model
#   to_streaming    process start until the pager is really moving bytes, since
#                   a request arriving mid-load has to wait behind it
B=$(dirname "$(readlink -f "$0")")
BIG=$(cat "$B/model_path.txt")
P="$B/prompt.txt"

[ -s "$P" ]  || { echo "prompt.txt empty - aborting";  exit 1; }
[ -s "$BIG" ] || { echo "model path empty - aborting"; exit 1; }
cd /home/atomic-germ/Projects/Guanaco-Test/build || exit 1

for CAP in 16G 30G; do
  for i in 1 2; do
    echo "########## cold start: 27B, cache dropped, ${CAP} cap, run $i ##########"
    python3 "$B/dropcache.py" "$BIG" 2>&1 | sed 's/^/  /'
    START=$(date +%s.%N)
    env LLAMA_ARG_LOAD_MODE=streaming LLAMA_ARG_GPU_PILL=1 LLAMA_ARG_KV_UNIFIED=0 \
        LLAMA_ARG_CACHE_TYPE_K=q8_0 LLAMA_ARG_CACHE_TYPE_V=q4_0 LLAMA_ARG_THREADS=6 \
        LLAMA_ARG_GUANACO_DENSE_BUDGET=7000 \
      systemd-run --user --quiet --scope -p MemoryMax=$CAP \
        ./bin/llama-cli -m "$BIG" -ngl 99 -c 1024 -n 4 -f "$P" -st --no-warmup 2>&1 \
      | grep -E "load_tensors: loading|resident subset|Prompt eval|Generation:" | head -3
    END=$(date +%s.%N)
    echo "  wall for 4 tokens: $(echo "$END - $START" | bc) s"
  done
done
echo "COLD START DONE"
