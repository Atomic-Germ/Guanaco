# Guanaco: Disk-Streaming for MoE Models on llama.cpp

**Core idea:** An MoE model (e.g. 122B params, 256 experts per layer, top-8) only activates a handful of experts per token. Guanaco exploits this by keeping only "hot" experts resident in RAM and streaming the rest from NVMe on demand — so a model that's larger than your total RAM/VRAM still runs.

## Architecture

**Build integration** (`CMakeLists.txt:36-88`): At configure time, CMake applies `.patch` files from `guanaco/patches/` onto `../llama.cpp` in-place (idempotent via `--dry-run` first), then builds llama.cpp as a subdirectory. `libllama.so` links against `libguanaco.so` — circular dependency is fine for shared libs.

**11 patches** inject Guanaco into llama.cpp at exactly 3 points:

1. **Model load** (`llama.cpp.patch`, `llama-model.cpp.patch`): After `load_tensors()` finishes, the patch iterates every fused expert tensor (e.g. `blk.5.ffn_up_exps.weight` — a tensor with 256 experts concatenated along the last dim). For each one, it records the mmap base address and per-expert slice geometry into the `SteppeLoader`. Critically, it does **not** redirect `t->data` — ggml still reads from the original file-backed mmap address. Streaming happens *in-place* via `madvise`.

2. **Context init** (`llama-context.cpp.patch`): Installs an `eval_callback` that fires after every `ffn_moe_topk-<layer>` node computes. The callback reads the router's selected expert IDs from the node's output tensor and calls `hook->on_router_computed()`.

3. **CLI flags** (`common-arg.cpp.patch`, `common-common.h.cpp.patch`): Adds `--guanaco-streaming`, `--guanaco-max-experts`, etc. to the arg parser.

## The streaming mechanism (`guanaco_steppeloader.cpp` — the real meat)

The key insight is that disk streaming is done **in-place on the mmap**:

- At load: all expert mmap pages are resident (the OS mapped the whole GGUF).
- `advise_experts_random()` (`:607`): `MADV_RANDOM` + `POSIX_FADV_RANDOM` on all expert regions — tells the kernel "don't sequential read-ahead, we access randomly."
- `prefetch_experts()` (`:1213`): When the router selects experts for a layer, for each selected expert: `madvise(addr, per_expert_bytes, MADV_WILLNEED)` — asks the kernel to page that slice in. Already-resident slices are no-ops.
- `evict_to_budget()` (`:828`): `MADV_DONTNEED` on cold slices — tells the kernel "drop these pages, I don't need them." The OS frees the RAM. Next access page-faults back from the GGUF file (same file-backed address, so the bytes are identical).
- Net effect: only `max_experts` slices per fused tensor are resident at any time. RAM usage is bounded by the knob, not the model size.

## Intelligent prefetching

- **Hot pinning** (`maybe_pin_hot_experts`, `:922`): Tracks per-expert hit counts with EWMA decay (`kHitDecay_ = 0.95`). After 200 layers of warmup, permanently pins the hottest experts per tensor — they stay resident forever.
- **PILOT cross-layer prefetch** (`pilot_prefetch_next_layer`, `:749`): Learns a per-tensor transition matrix `(from_expert -> to_expert)` across layers. When layer N's router fires, it predicts layer N+1's likely experts and `MADV_WILLNEED`s them early — hiding disk latency behind compute. Pruned by `pilot_mass` (default 0.9) to skip the long tail.
- **Imatrix seeding** (`load_imatrix_prior`, `:967`): Loads a sibling `<model>.imatrix.gguf` (calibration data) to seed hit counts before token 0, so the cold-start pin pass picks the right experts immediately.
- **Dynamic budgets** (`recompute_budgets`, `:885`): Rebalances per-tensor slot allocations proportional to distinct-expert demand. Busy tensors get more slots; idle ones get fewer. Total stays within `max_experts * n_tensors`.
- **Two-pass eviction** (`:853`): PASS A only evicts cold slices (protecting hot ones); PASS B falls back to pure LRU if still over budget.

## I/O backends

- **io_uring** (`guanaco_io_uring.cpp`): Batched async reads — submits all slices for a layer in one `io_uring` batch, reaps completions together. Lets the kernel dispatch NVMe reads in parallel.
- **madvise fallback**: When io_uring is unavailable or disabled, `MADV_WILLNEED`/`MADV_DONTNEED` on the mmap drives everything through normal page faults.

## Sharded GGUF support

`build_shard_paths()` (`:85`) globs sibling shards (`*-00001-of-00004.gguf` etc.), and `read_gguf_metadata_ctx()` (`:148`) reads only the metadata region via pread (never mmaps the full tensor data), so discovering experts across shards costs negligible RAM.

## Dense model safety

If `num_experts <= 1` (not MoE), `enabled_ = false` and all methods become no-ops — ggml's normal mmap runs untouched (`:503`).

"This is a genuinely elegant approach — instead of copying expert weights into separate buffers (which doubles memory), it lets the OS page cache be the cache, using `madvise` hints to control what's resident. That's why your 139B model runs in <10GB." -- Qwen 3.7 Plus
