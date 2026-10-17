#!/usr/bin/env bash

cmake -B build -DCMAKE_C_FLAGS="-march=znver5 -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512cd" -DCMAKE_CXX_FLAGS="-march=znver5 -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512cd" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
