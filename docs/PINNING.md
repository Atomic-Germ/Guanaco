# Guanaco hot-expert pinning and the GUANACO_MAX_EXPERTS knob

This document explains how Guanaco decides which MoE experts stay resident in
RAM (the "hot" experts) versus which are streamed from NVMe on demand, and how
to tune that behavior for different workloads.

## Background: why pinning matters

MoE models (Mixtral, Qwen3-MoE, DeepSeek, ...) route each token to only a few
experts out of many. For a 14B-A3B model there are 90 experts per layer but the
router picks 8. Most of the expert weights are never used for a given token, so
keeping all of them in RAM is wasteful. Guanaco instead:

1. Memory-maps each fused expert tensor into a sparse slab. Nothing is resident
   until it is read.
2. After the router picks experts for a token, only those expert slices are
   pread (or io_uring-read) from the GGUF into the slab, just in time for
   the mul_mat_id compute.
3. Tracks how often each expert is selected and permanently pins the hottest
   ones into RAM so they are never re-read from disk.

Pinned experts cost zero per-token disk I/O. Cold experts stream on demand.
This is the "HerdCache" from the project design: a small, fast, rolling window
of resident experts named after the open plains where guanacos roam.

## What GUANACO_MAX_EXPERTS controls

GUANACO_MAX_EXPERTS is the pin budget: the maximum number of experts kept
resident per fused expert tensor (per layer, per gate/up/down projection). It is
the single most important tuning knob for the RAM/disk-latency tradeoff.

- Higher value -> more experts pinned -> higher pin-cache hit rate -> fewer
  disk reads -> faster, more stable token times, but more RAM used.
- Lower value -> less RAM, but more experts must stream from NVMe each step,
  which shows up as the slow (disk-bound) tail in prompt-processing times.

The metric to watch is the pin-cache hit rate, printed periodically:

```
[Guanaco HerdCache] pin cache: 57% resident (14% hot-pinned), 14043/24480 expert accesses, pinned=1728 across 120 tensors
```

- `resident %` = share of router-selected expert accesses that needed no disk
  read at all (pinned + already-loaded slices).
- `hot-pinned %` = share served directly by a permanently pinned expert.
- `pinned=N` = total number of experts currently pinned across all tensors.

If `resident %` is low and your workload is latency-sensitive, raise
GUANACO_MAX_EXPERTS. If RAM is tight and `resident %` is already high, you can
lower it to free memory with little speed cost.

## How to set it

Environment variable (read at model load):

```sh
GUANACO_STREAMING=1 GUANACO_MAX_EXPERTS=16 ./llama-cli -m model.gguf ...
```

Accepted values:

- GUANACO_MAX_EXPERTS unset -> auto default of 8 experts per tensor.
- GUANACO_MAX_EXPERTS=N (N > 0) -> pin at most N experts per tensor.
- The model-loading API also takes guanaco_max_experts; a value of -1 means
  "auto" (resolves to 8). The env var overrides both.

RAM impact is roughly: N * num_layers * 3 projections * per_expert_bytes.
For the Qwen3.6-14B-A3B Q6_K test model, one expert slice is ~0.8 MB, so:

- GUANACO_MAX_EXPERTS=8  -> ~960  pinned experts -> ~0.8 GB RAM for pinned weights.
- GUANACO_MAX_EXPERTS=16 -> ~1920 pinned experts -> ~1.6 GB RAM for pinned weights.
- GUANACO_MAX_EXPERTS=90 -> all experts pinned -> equivalent to loading the
  whole FFN into RAM (no streaming benefit, but no disk reads either).

## Two sources of the hot set

Guanaco seeds the pinned set from two signals, in order:

1. imatrix prior (cold start). If a sibling file <model>.gguf.imatrix.gguf
   exists next to the model, Guanaco reads the per-expert selection counts from
   it and pins the calibration-hot experts before the first token. This means
   pinning is effective immediately instead of waiting for routing statistics to
   warm up. Generate one with llama.cpp's llama-imatrix tool:
   ```sh
   ./llama-imatrix -m model.gguf -f calib.txt -o model.gguf.imatrix.gguf
   ```
2. Runtime routing frequency. As tokens are generated, record_routing
   maintains an EWMA "windowed hotness" per expert. After a short warmup the
   hottest experts (by runtime frequency) are additionally pinned, adapting to
   the actual prompt/domain even when no imatrix is present.

Both feed the same pinned[] residency set; prefetch_experts skips pinned
experts, so they never incur a disk read.

## Tuning examples by use case

These are starting points. Watch the `pin cache:` line and adjust.

### Focused coding session (narrow domain)
The model stays within a small vocabulary of experts (API shapes, a language's
syntax, your repo's patterns). Routing is concentrated, so a small budget pins
the relevant experts and most tokens hit cache.

```sh
GUANACO_STREAMING=1 GUANACO_MAX_EXPERTS=6 ./llama-cli -m model.gguf -p "Refactor utils.ts to use async iterators"
```
Expect a high resident % even at a low budget. Good place to save RAM.

### General-purpose chat (broad, shifting topics)
Conversation ranges across many subjects, so expert usage is spread out. A
larger budget keeps more experts hot and absorbs topic switches without hitting
disk.

```sh
GUANACO_STREAMING=1 GUANACO_MAX_EXPERTS=16 ./llama-cli -m model.gguf
```
If you see the resident % drop during long, wide-ranging chats, raise it
further (24-32) until it stabilizes, then stop - more RAM past that point buys
little.

### Long-context / novel-writing agent (your OS-from-scratch workload)
A long, coherent single task (writing an OS, a novel) builds up very stable
routing over thousands of tokens. Token generation gets faster as more experts
pin; the slow tail is prompt processing when the router reaches an expert that
is not yet resident. Pair a generous budget with an imatrix prior so the hot set
is correct from the start:

```sh
# produce the prior once
./llama-imatrix -m model.gguf -f domain_corpus.txt -o model.gguf.imatrix.gguf
# run with a large pin budget
GUANACO_STREAMING=1 GUANACO_MAX_EXPERTS=24 ./llama-cli -m model.gguf -p "Write a bootable OS..."
```
The wide variance you observed in prompt-processing time (.1s vs 120s) is
exactly the pinned-vs-streamed difference: once the working set is pinned, those
slow steps disappear. A larger budget (or a good imatrix) is what flattens it.

### Memory-constrained machine (must fit in a small RAM budget)
If you cannot spare ~1-2 GB for pinned weights, drop the budget and accept more
streaming. Generation stays correct; only the disk-bound tail grows.

```sh
GUANACO_STREAMING=1 GUANACO_MAX_EXPERTS=2 ./llama-cli -m model.gguf
```
Watch `pin cache:` - if resident % stays very low, the NVMe is doing most of the
work and you are effectively in pure disk-streaming mode.

### Max throughput (RAM is abundant)
Pin everything; Guanaco behaves like a normally-loaded model with no streaming
overhead.

```sh
GUANACO_STREAMING=1 GUANACO_MAX_EXPERTS=90 ./llama-cli -m model.gguf
```

## Relationship to model architecture

The right budget depends more on routing behavior than on model family.
Models with sharp, concentrated routing (few experts dominate) need a small
budget. Models with flat, diffuse routing (many experts used evenly) need a
larger budget to reach a high hit rate, or they will stream a lot. The
`pin cache:` metric tells you which regime you are in - tune to the workload,
not the model name.

## Notes

- GUANACO_MAX_EXPERTS caps pinning per tensor; it does not limit how many
  cold experts can stream. Streaming is unbounded and always available.
- Pinned experts are chosen by either the imatrix prior (cold start) or runtime
  EWMA hotness (warmup + adaptation). Raising the budget lets more of the hot
  set stay resident; it never forces unused experts into RAM.
- The metric is cumulative for the process; resident % trends up as the hot set
  stabilizes, which is why long, coherent tasks speed up over time.
