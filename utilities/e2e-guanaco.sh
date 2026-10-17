#!/bin/bash
# End-to-end test of the user-facing flow:
#   git clone <Guanaco> && cd Guanaco && cmake -B build -DGGML_VULKAN=ON && cmake --build build -j6
# We simulate the clone by copying the current working tree (tracked files only,
# no llama.cpp/ and no build dirs), then let CMake clone the pinned llama.cpp.
set -euo pipefail

SRC=/home/atomic-germ/Projects/Guanaco-bak
DST=/tmp/opencode/guanaco-e2e

rm -rf "$DST"
mkdir -p "$DST"
tar -C "$SRC" \
    --exclude=./llama.cpp \
    --exclude=./build \
    --exclude=./build-vulkan \
    --exclude=./.git \
    -cf - . | tar -C "$DST" -xf -

cd "$DST"
echo "### configure (fresh llama.cpp clone + patches) ###"
cmake -B build -DGGML_VULKAN=ON
echo "### build ###"
cmake --build build -j6
echo "### E2E OK ###"
ls -la build/bin/llama-server build/bin/libguanaco.so
