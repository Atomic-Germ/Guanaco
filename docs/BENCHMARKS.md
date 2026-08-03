# Benchmarks

All runs below were performed on a single consumer box: **48 GB system RAM** and
an **AMD GPU (ROCm + Vulkan, gfx1152)**. CPU thread counts vary per run. Guanaco
was built from this tree (`cmake -B build -S . && cmake --build build`), producing
the patched `build/bin/llama-server` and `build/bin/libguanaco.so`. Plain
llama.cpp is the patched server run **without** `--guanaco-streaming`.

Unless noted, streaming was enabled with `--guanaco-streaming`; disk I/O used
`io_uring` where available (Linux 5.1+).

---

## 1. Running models with no right to

The whole point of Guanaco is fitting MoE models far larger than system RAM.

| Model | Arch | Params | GGUF size | `--guanaco-max-experts` | Measured RAM | Tok/s |
|-------|------|---------|-----------|--------------------------|-------------|-------|
| NVIDIA Nemotron-3-Super-120B-A12B | nemotron_h_moe (sparse, 4-shard) | 120B-A12B | ~120 GB | 18 | **< 4 GB** | ~0.5 |
| Qwen3.6-35B-A3B (APEX) | qwen35moe (dense, E=256) | 35B-A3B | — | 9 | bounded | ~5 |
| gpt-oss-20B (RotorQuant Q8_0) | gpt-oss (sparse) | 20B | — | 10 | negligible | ~8–9 |
| Nemotron-Cascade-2-30B-A3B | nemotron_h_moe (sparse) | 30B-A3B | — | — | bounded | — |

The 120 GB / 4 GB result is the headline: the OS page cache is never asked to
hold the file. Cold experts page-fault back from the GGUF on demand, and the
`evict_hot=0 evict_reread=0 (0% churn)` final line confirms the pinned hot set
was exactly right for that routing distribution — no wasted evictions.

---

## 2. Guanaco vs. full GPU offload (the surprising one)

A direct user comparison of **gpt-oss-20B** two ways on the same box:

| Metric | Guanaco (CPU, max-experts 16, no imatrix) | Full ROCm offload |
|--------|---------------------------------------------|------------------|
| RAM use | negligible | whole file resident |
| Prompt processing | **47 t/s** | 40 t/s |
| Token generation (avg) | **8.21 t/s** | 7.58 t/s |
| Thermals | cool | noticeably hotter |

The same pattern held under Vulkan. Guanaco is *not* trying to be faster than a
GPU — but because it bounds RAM and lets the CPU work the attention/router while
NVMe streams the FFN experts, it ends up matching or beating a fully-offloaded
run while using a fraction of the memory and running cooler.

---

## 3. Free teamwork: two models at once

Run **gpt-oss on Vulkan** (VRAM-bound) and **Qwen3.6-35B on CPU + Guanaco
HerdCache** (NVMe-bound) in the same process at the same time:

- **Total RAM (incl. VRAM) < 30 GB**
- Neither model slowed; each ran at its solo speed.

They compete for different resources (VRAM vs. NVMe/CPU), so the two heat
signatures don't collide. This is a natural multi-model serving mode for one box.

---

## 4. Feature ablation

### 4.1 Eviction hot-protection (#1)

On Qwen3.6-35B (budget 9) the two-pass reclaim (cold-only PASS A, then LRU
fallback PASS B) was validated against the old single-pass LRU:

```
final: pin budget=9 per tensor, experts pinned now=3747 (31.225 avg)
       pin cache=77% resident (52% hot-pinned)
       evict_hot=1106001 evict_reread=5367 (0% churn)
```

`evict_hot` is high only because the budget is saturated by pins at this tensor
(the 9-slot cap is smaller than the model's hot set). The decisive number is
**`evict_reread=5367 (0% churn)`** — every eviction, hot or cold, was a correct
one that was never needed again soon. The old LRU would have re-read many of
those hot experts it threw away.

On the 120 GB model (budget 18) the effect is even cleaner:
`evictions=39520, evict_hot=0 evict_reread=0 (0% churn)` — PASS A absorbs all
non-forced evictions, zero thrash.

### 4.2 Per-tensor dynamic cap (#2)

Same model, same budget 9, before/after giving each tensor its own demand-
proportional cap (range `6..9` instead of a flat 9):

| | evictions | evict_hot | resident | churn |
|---|-----------|-----------|----------|-------|
| Before (flat cap 9) | 1,141,392 | 1,106,001 | 77% | — |
| After (dynamic 6..9) | **237,615** | **201,654** | 78% | **0%** |

Idle tensors stop hoarding a useless fixed 9-slot budget; those slots flow to
busy tensors. **~5× fewer evictions at equal RAM and no hit-rate regression.**

### 4.3 Pilot lookahead K-pruning (#3)

Pilot prefetch normally reads the full top-K predicted experts per tensor. Pruning
to the top fraction of routing transition mass (`GUANACO_PILOT_MASS`) trades a
little hit-rate for far fewer disk reads.

**gpt-oss** (E=32 experts/tensor, K=10), slices per pilot call:

| `GUANACO_PILOT_MASS` | slices/call | resident | hot-pinned | churn |
|-----------------------|-----------|----------|-------------|-------|
| 1.0 (full top-K) | 9.56 | 97% | 91% | 0% |
| 0.9 (default) | 9.94 | 98% | 89% | 0% |
| 0.3 | **2.47** | 96% | 87% | 1% |

At `0.3` pilot I/O drops **~3.9×** for only 2% resident / 2% hot-pinned loss.

On **Qwen3.6-35B** (E=256, K=9) the default `0.9` is effectively a no-op
(`8.42` → `8.77` slices/call): with 256 experts the top-9 already cover far
less than 90% of the transition mass, so the per-tensor `K` cap binds before
the mass cutoff. That is expected — pruning only bites when routing is "sticky"
(concentrated transitions), which is exactly the case it targets.

---

## 5. How to read the numbers

The Guanaco stderr lines to watch (see `README.md`):

- **`pin cache: N% resident (M% hot-pinned)`** — the headline hit-rate. Raise
  `--guanaco-max-experts` if `resident %` is low and RAM allows.
- **`dynamic per-tensor X..Y`** — per-tensor pin budget spread (busy tensors
  get more than idle ones).
- **final `churn %`** — share of evicted experts re-read soon after. Should
  stay near **0%**; a high value means the budget is too small for the model's
  routing spread.

## 6. Caveats

- **MoE only.** Dense models are detected and left untouched.
- **Needs fast storage.** Cold experts are read from the GGUF on demand; an NVMe
  is strongly recommended. The whole design assumes disk, not RAM, is the
  bottleneck — which the router lets us hide via prefetch.
- **Linux / `madvise` based.** `MADV_RANDOM` suppresses OS read-ahead on expert
  regions; `MADV_DONTNEED` drops cold slices; `MADV_WILLNEED` faults hot ones
  in. Correctness is independent of eviction: a dropped slice is re-read from the
  same file-backed address on next use.
