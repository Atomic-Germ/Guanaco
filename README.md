# Guanaco (llama.cpp delta)

Guanaco is a disk-streaming add-on for **llama.cpp MoE models**. It assumes you
already know how to build and run llama.cpp; this document only covers what is
*different*.

The one idea: a Mixture-of-Experts model routes each token to a handful of
experts out of many, so most expert weights are idle at any moment. Plain
llama.cpp memory-maps the whole GGUF and relies on the OS page cache to keep
the working set in RAM. Guanaco instead **keeps only a small set of "hot"
experts pinned in RAM and streams the rest from NVMe on demand**, so a model
that would never fit in system memory runs anyway — bounded by a knob you
control, not by the file size.

<img width="2880" height="1854" alt="image" src="https://github.com/user-attachments/assets/65311726-6a53-4cb8-abcc-5725d447269d" />
Test system is a Framework 13 AMD Ryzen 340 AI with 48GB of total ram/vram shared, 6 cores on the cpu. OS is Fedora 45 Rawhide, kernel 7.2. Both models; Qwen3.5 122B A10B and Qwen3.6 35B A3B are running concurrently on that system. Ram use is largely context, and the gpu is unused for now. Example is built against llama.cpp build 10068.

Dense (non-MoE) models are untouched currently (maybe forever: Guanaco detects no expert tensors and
does nothing.

## Build

Guanaco is a CMake project that **patches and builds llama.cpp as a
subdirectory**. The patches add the `--guanaco-streaming` flag and the router
hook; they are applied automatically at configure time, so a plain llama.cpp
checkout next to this repo is all you need.

```sh
# llama.cpp must be a sibling of this guanaco/ directory
ls ../llama.cpp        # the upstream checkout to be patched in place

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

## Quick start (CPU / RAM-bound)

The usual case: you want a big MoE model on CPU with limited RAM.

```sh
./build/bin/llama-server \
    -m model.gguf \
    -ngl 0                       `# CPU only; Guanaco owns the expert streaming` \
    --guanaco-streaming          `# enable disk streaming` \
    --guanaco-max-experts 12     `# hot experts pinned per tensor (tune for RAM)`
```

On a 48 GB box this runs a 120 GB MoE in well under 4 GB of RAM, with the
cold experts page-faulting back from the GGUF as the router selects them.

It also coexists with GPU offload: e.g. run one model fully offloaded to a
ROCm/Vulkan device **and** a second model on CPU with `--guanaco-streaming`
at the same time — they compete for different resources (VRAM vs. NVMe) and
each runs at its native speed.

## Enabling and tuning

| Flag | Env override | Default | Meaning |
|------|-------------|---------|---------|
| `--guanaco-streaming` | `GUANACO_STREAMING=1` | off | Turn on expert disk streaming. |
| `--guanaco-max-experts N` | `GUANACO_MAX_EXPERTS=N` | `-1` (auto → 8) | Hot experts pinned **per fused tensor**. Higher = more RAM, fewer disk reads; lower = less RAM, more streaming. |
| (n/a) | `GUANACO_PILOT=0` | on | Cross-layer lookahead prefetch. Hints the next layer's likely experts while the current block computes. A wrong guess costs at most a wasted read. |
| (n/a) | `GUANACO_PILOT_MASS=0.9` | `0.9` | Pilot reads only the top fraction of cumulative routing transition mass. Lower (e.g. `0.3`) trades a little hit-rate for far fewer pilot I/Os. `1.0` = full top-K. |
| (n/a) | `GUANACO_IMATRIX=0` | on | Seed the hot set from a sibling `<model>.imatrix.gguf` before the first token. |
| (n/a) | `GUANACO_IO_URING=0` | on (if available) | Force the thread-pool I/O backend instead of `io_uring`. |

`GUANACO_MAX_EXPERTS` (env) overrides the `--guanaco-max-experts` flag, which
overrides the auto default.

Deep dive on the pinning knob and the resident/hit-rate tradeoff: see
[`docs/PINNING.md`](docs/PINNING.md).

Measured results (including the 120 GB-in-4 GB run and the Guanaco-vs-GPU-offload
comparison): see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

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

## Notes & caveats

- **Requires an MoE architecture** (Mixtral, DeepSeek, Nemotron-H, Qwen-MoE,
  gpt-oss, LFM2, afmoe, …). Sparse-MoE layouts (MoE blocks at non-contiguous
  layers) are handled.
- **Needs fast storage.** Cold experts are read from the GGUF on demand; an
  NVMe is strongly recommended. The whole point is that disk, not RAM, becomes
  the bottleneck — which is fine because the router lets us prefetch.
- **Linux / `madvise` based.** `MADV_RANDOM` stops the OS from doing
  sequential read-ahead on expert regions; `MADV_DONTNEED` drops cold slices;
  `MADV_WILLNEED` faults hot ones in. Behavior is correct regardless: a dropped
  slice is re-read from the same file-backed address on next use.
- **It is a patch on top of llama.cpp.** Build through this project's CMake (it
  patches `../llama.cpp` in place) rather than building llama.cpp directly, or
  the `--guanaco-streaming` flag and hook will be absent.
- **Sharded GGUFs** (e.g. `-0000N-of-0000M.gguf`) are supported; the manifest
  is parsed per shard and the imatrix lookup tolerates the shard suffix.
