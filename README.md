# Guanaco: Disk-Streaming for MoE Models on llama.cpp

**Core idea:** An MoE model (e.g. 122B params, 256 experts per layer, top-8) only
activates a handful of experts per token. Guanaco exploits this by keeping only
"hot" experts resident in RAM and streaming the rest from NVMe on demand — so a
model that's larger than your total RAM/VRAM still runs.

Plain llama.cpp memory-maps the whole GGUF and relies on the OS page cache to
keep the working set in RAM. Guanaco instead **keeps only a small set of "hot"
experts pinned in RAM and streams the rest from NVMe on demand**, bounded by a
knob you control, not by the file size.

<img width="2880" height="1854" alt="image" src="https://github.com/user-attachments/assets/65311726-6a53-4cb8-abcc-5725d447269d" />

(tested with `systemd-run --user --scope -p MemoryMax=15000000000 -p MemorySwapMax=0 llama-server -c 2048 --spec-type none -hf jamiefutch/Qwen3.5-122B-A10B-MXFP4_MOE-MTP-GGUF:MERGED`)
<img width="2852" height="1644" alt="Screenshot From 2026-08-13 12-01-29" src="https://github.com/user-attachments/assets/2c7a381e-7bfb-4d4b-b1ae-8174c7ef7f67" />

Test system is a Framework 13 AMD Ryzen 340 AI with 48GB of total ram/vram shared, 6 cores on the cpu. OS is Fedora 45 Rawhide, kernel 7.2. Both models; Qwen3.5 122B A10B and Qwen3.6 35B A3B are running concurrently on that system. Ram use is largely context, and the gpu is unused for now.

Dense (non-MoE) models are untouched currently (maybe forever): Guanaco detects
no expert tensors and does nothing.

## Repo layout

This repository is just three things:

- **`patches/`** — the Guanaco delta on llama.cpp, applied automatically at
  configure time
- **a build system** (`CMakeLists.txt`) that patches and builds llama.cpp as a
  subdirectory, plus the Guanaco streaming library sources in `src/`
- **`llama.cpp/`** — a git submodule (upstream `ggml-org/llama.cpp`); the
  patches are applied to its working tree, never committed to it

## Build

```sh
git submodule update --init     # fetch the llama.cpp submodule

cmake -B build -S .
cmake --build build -j$(nproc)
```

This produces a **patched** server and the streaming library:

```
build/bin/llama-server      # patched; auto-enables Guanaco when requested
build/bin/libguanaco.so
```

`io_uring` is used automatically when available (Linux 5.1+); otherwise it
falls back to a thread pool. Either way the behavior is identical.

## Architecture

**Build integration** (`CMakeLists.txt:36-64`): At configure time, CMake
applies `.patch` files from `patches/` onto the `llama.cpp` submodule in-place
(idempotent via `--dry-run` first), then builds llama.cpp as a subdirectory.
`libllama.so` links against `libguanaco.so` — circular dependency is fine for
shared libs.

**12 patches** inject Guanaco into llama.cpp at exactly 3 points:

1. **Model load** (`llama.cpp.patch`, `llama-model.cpp.patch`): After
   `load_tensors()` finishes, the patch iterates every fused expert tensor
   (e.g. `blk.5.ffn_up_exps.weight` — a tensor with 256 experts concatenated
   along the last dim). For each one, it records the mmap base address and
   per-expert slice geometry into the `SteppeLoader`. Critically, it does **not**
   redirect `t->data` — ggml still reads from the original file-backed mmap
   address. Streaming happens *in-place* via `madvise`.
2. **Context init** (`llama-context.cpp.patch`): Installs an `eval_callback`
   that fires after every `ffn_moe_topk-<layer>` node computes. The callback
   reads the router's selected expert IDs from the node's output tensor and
   calls `hook->on_router_computed()`.
3. **CLI flags** (`common-arg.cpp.patch`, `common-common.h.cpp.patch`): Adds
   `--load-mode streaming` (plus `--guanaco-max-experts`, etc.) to the arg
   parser. `--load-mode streaming` is the real flag; `--guanaco-streaming` /
   `-gs` is kept as a deprecated alias.

## Quick start (CPU / RAM-bound)

The usual case: you want a big MoE model on CPU with limited RAM.

```sh
./build/bin/llama-server \
    -m model.gguf \
    -ngl 0                       `# CPU only; Guanaco owns the expert streaming` \
    --load-mode streaming       `# enable expert disk streaming` \
    --guanaco-max-experts 12     `# hot experts pinned per tensor (tune for RAM)`
```

On a 48 GB box this runs a 120 GB MoE in well under 4 GB of RAM, with the
cold experts page-faulting back from the GGUF as the router selects them.

It also coexists with GPU offload: e.g. run one model fully offloaded to a
ROCm/Vulkan device **and** a second model on CPU with `--load-mode streaming`
at the same time — they compete for different resources (VRAM vs. NVMe) and
each runs at its native speed.

## The streaming mechanism (`src/guanaco_steppeloader.cpp` — the real meat)

The key insight is that disk streaming is done **in-place on the mmap**:

- At load: all expert mmap pages are resident (the OS mapped the whole GGUF).
- `advise_experts_random()` (`:607`): `MADV_RANDOM` + `POSIX_FADV_RANDOM` on
  all expert regions — tells the kernel "don't sequential read-ahead, we access
  randomly."
- `prefetch_experts()` (`:1213`): When the router selects experts for a layer,
  for each selected expert: `madvise(addr, per_expert_bytes, MADV_WILLNEED)` —
  asks the kernel to page that slice in. Already-resident slices are no-ops.
- `evict_to_budget()` (`:828`): `MADV_DONTNEED` on cold slices — tells the
  kernel "drop these pages, I don't need them." The OS frees the RAM. Next
  access page-faults back from the GGUF file (same file-backed address, so the
  bytes are identical).
- Net effect: only `max_experts` slices per fused tensor are resident at any
  time. RAM usage is bounded by the knob, not the model size.

## Enabling and tuning

| Flag | Env override | Default | Meaning |
|------|-------------|---------|---------|
| `--load-mode streaming` | `LOAD_MODE=streaming` | `mmap` | Turn on expert disk streaming. The streaming mode is implemented on top of mmap, so `use_mmap` is implied. The `--guanaco-streaming` / `-gs` flag is kept as a deprecated alias. |
| `--guanaco-max-experts N` | `GUANACO_MAX_EXPERTS=N` | `-1` (auto → 8) | Hot experts pinned **per fused tensor**. Higher = more RAM, fewer disk reads; lower = less RAM, more streaming. |
| (n/a) | `GUANACO_PILOT=0` | on | Cross-layer lookahead prefetch. Hints the next layer's likely experts while the current block computes. A wrong guess costs at most a wasted read. |
| (n/a) | `GUANACO_PILOT_MASS=0.9` | `0.9` | Pilot reads only the top fraction of cumulative routing transition mass. Lower (e.g. `0.3`) trades a little hit-rate for far fewer pilot I/Os. `1.0` = full top-K. |
| (n/a) | `GUANACO_IMATRIX=0` | on | Seed the hot set from a sibling `<model>.imatrix.gguf` before the first token. |
| (n/a) | `GUANACO_IO_URING=0` | on (if available) | Force the thread-pool I/O backend instead of `io_uring`. |

`GUANACO_MAX_EXPERTS` (env) overrides the `--guanaco-max-experts` flag, which
overrides the auto default. The `--guanaco-streaming` / `-gs` / `-ngs` /
`--no-guanaco-streaming` flags and the `GUANACO_STREAMING` env var are
deprecated aliases for `--load-mode streaming` (and the default `--load-mode
mmap`); scripts should migrate to `--load-mode streaming`.

Deep dive on the pinning knob and the resident/hit-rate tradeoff: see
[`docs/PINNING.md`](docs/PINNING.md).

Measured results (including the 120 GB-in-4 GB run and the Guanaco-vs-GPU-offload
comparison): see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Intelligent prefetching

- **Hot pinning** (`maybe_pin_hot_experts`, `:922`): Tracks per-expert hit
  counts with EWMA decay (`kHitDecay_ = 0.95`). After 200 layers of warmup,
  permanently pins the hottest experts per tensor — they stay resident forever.
- **PILOT cross-layer prefetch** (`pilot_prefetch_next_layer`, `:749`): Learns a
  per-tensor transition matrix `(from_expert -> to_expert)` across layers. When
  layer N's router fires, it predicts layer N+1's likely experts and
  `MADV_WILLNEED`s them early — hiding disk latency behind compute. Pruned by
  `pilot_mass` (default 0.9) to skip the long tail.
- **Imatrix seeding** (`load_imatrix_prior`, `:967`): Loads a sibling
  `<model>.imatrix.gguf` (calibration data) to seed hit counts before token 0,
  so the cold-start pin pass picks the right experts immediately.
- **Dynamic budgets** (`recompute_budgets`, `:885`): Rebalances per-tensor slot
  allocations proportional to distinct-expert demand. Busy tensors get more
  slots; idle ones get fewer. Total stays within `max_experts * n_tensors`.
- **Two-pass eviction** (`:853`): PASS A only evicts cold slices (protecting hot
  ones); PASS B falls back to pure LRU if still over budget.

## I/O backends

- **io_uring** (`src/guanaco_io_uring.cpp`): Batched async reads — submits all
  slices for a layer in one `io_uring` batch, reaps completions together. Lets
  the kernel dispatch NVMe reads in parallel.
- **madvise fallback**: When io_uring is unavailable or disabled,
  `MADV_WILLNEED`/`MADV_DONTNEED` on the mmap drives everything through normal
  page faults.

## Sharded GGUF support

`build_shard_paths()` (`:85`) globs sibling shards (`*-00001-of-00004.gguf`
etc.), and `read_gguf_metadata_ctx()` (`:148`) reads only the metadata region
via pread (never mmaps the full tensor data), so discovering experts across
shards costs negligible RAM.

## What you'll see (observability)

Guanaco logs to stderr with the `[Guanaco HerdCache]` / `[Guanaco Storage]`
prefix. The lines worth watching:

```
[Guanaco Storage] Model config: qwen35moe, experts=256, top_k=8, layers=40, hidden=2048
[Guanaco Storage] Parsed 180 expert tensors across 41 layers (from 1 shard(s))
[Guanaco HerdCache] pin budget=9 (dynamic per-tensor 6..9), experts pinned now=3372 ...
```

- **`dynamic per-tensor X..Y`** — the per-tensor pin budget is *not* uniform.
  Busy tensors (many distinct experts) get more slots; idle ones get fewer, so
  the same total RAM is spent where it is actually used.
- **Periodic pin-cache line:**

  ```
  [Guanaco HerdCache] pin cache: 79% resident (52% hot-pinned),
      1470657/1861566 expert accesses, distinct experts pinned=3372 across 120 tensors
  ```

  - `resident %` — share of router-selected experts that needed **no disk read**
    (pinned + already-warm). This is the headline number.
  - `hot-pinned %` — share served directly by a permanently pinned expert.
  - Low `resident %` and RAM to spare ⇒ raise `--guanaco-max-experts`.
- **Final summary (on shutdown):**

  ```
  final: pin budget=9 (dynamic per-tensor 6..9), experts pinned now=3372 ...
  pin cache=79% resident (52% hot-pinned), 1863438 expert accesses over 74427 prefetch calls,
  evictions=416679, pilot: 36667 calls / 308618 slices,
  evict_hot=380364 evict_reread=1795 (0% churn), imatrix prior used
  ```

  - `evict_hot` / `evict_reread` and the **`churn %`** tell you how often an
    evicted expert was needed again soon. It should stay near **0%** — a high
    churn means the budget is too small for the model's routing spread.

## Dense model safety

If `num_experts <= 1` (not MoE), `enabled_ = false` and all methods become
no-ops — ggml's normal mmap runs untouched (`:503`).

## Notes & caveats

- **Requires an MoE architecture** (Mixtral, DeepSeek, Nemotron-H, Qwen-MoE,
  Gemma 4, gpt-oss, LFM2, afmoe, …). Sparse-MoE layouts (MoE blocks at
  non-contiguous layers) are handled.
- **Needs fast storage.** Cold experts are read from the GGUF on demand; an
  NVMe is strongly recommended. The whole point is that disk, not RAM, becomes
  the bottleneck — which is fine because the router lets us prefetch.
- **Linux / `madvise` based.** `MADV_RANDOM` stops the OS from doing sequential
  read-ahead on expert regions; `MADV_DONTNEED` drops cold slices;
  `MADV_WILLNEED` faults hot ones in. Behavior is correct regardless: a dropped
  slice is re-read from the same file-backed address on next use.
- **It is a patch on top of llama.cpp.** Build through this project's CMake (it
  patches the `llama.cpp` submodule in place) rather than building llama.cpp
  directly, or the `--load-mode streaming` flag and hook will be absent.
- **Sharded GGUFs** (e.g. `-0000N-of-0000M.gguf`) are supported; the manifest
  is parsed per shard and the imatrix lookup tolerates the shard suffix.

"This is a genuinely elegant approach — instead of copying expert weights into
separate buffers (which doubles memory), it lets the OS page cache be the cache,
using `madvise` hints to control what's resident. That's why your 139B model
runs in <10GB." -- Qwen 3.7 Plus
