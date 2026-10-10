# PLAN: ToMoE-style streaming pseudo-experts for dense models

Status: draft for review. No code written yet.

## Summary

Guanaco streams MoE expert weights from disk and keeps only a hot subset in RAM.
Dense models have no experts, so Guanaco currently disables itself on them
(`src/guanaco_steppeloader.cpp:505`).

This project is an experiment: turn the MLP intermediate dimension of a dense
model into a set of pseudo-experts, route each token to a subset, and stream the
rest from disk. This mirrors the structure of ToMoE (dense MLP -> top-1 MoE along
the middle dimension) but without any weight training. Selection is guided by an
importance matrix (imatrix) that must ship next to the model.

## Hard requirements

- Input must be a dense GGUF with a sibling imatrix file. No imatrix, no run.
- We only drop/stream channels, we never edit or fine-tune model weights.
- CPU path only. Streaming is `-ngl 0`. No GPU offload of pseudo-experts.
- Streaming must not trigger the earlier GPU-placeholder heap corruption
  (register host tensors only; keep the `ggml_backend_buffer_is_host` guard).

## Non-goals

- Training the ToMoE hypernetwork, or running ToMoE Pipeline A/B.
- Support for every dense architecture. Start with one arch and one model.
- Any permanent re-quantization of the source model.

## Test target

Repo `mradermacher/Qwen3.5-27B-uncensored-heretic-v1-i1-GGUF`, revision
`9d27b916c9175b4ad0e6e1833f5e5fd063ea1718`:

- `Qwen3.5-27B-uncensored-heretic-v1.i1-Q4_K_M.gguf` (16.6 GB, dense)
- `Qwen3.5-27B-uncensored-heretic-v1.imatrix.gguf` (13.6 MB, required)

This is a dense Qwen3.5 27B, closely related to the Qwen3.5 MoE family Guanaco
already handles, with a matching imatrix in the same repo.

## Data we already have

The llama.cpp imatrix stores, per dense weight tensor, an `in_sum2` vector. For
`blk.N.ffn_down.weight` that vector has length `n_ff` and holds the sum of
squared intermediate activations per channel over the calibration set
(`llama.cpp/tools/imatrix/imatrix.cpp:640-660`). That is a per-channel salience
signal for the exact dimension we want to split. Guanaco already loads imatrix
files through `common_imatrix_load` (`src/imatrix-loader.cpp`).

The current Guanaco imatrix use is MoE-specific: it reads per-expert `counts` to
seed hot pins (`load_imatrix_prior`, `src/guanaco_steppeloader.cpp:969`). The
dense experiment needs the other axis: per-channel `sums`.

## Measured facts on the test target

Read from the downloaded imatrix (13.6 MB, 992 tensors, 496 base tensors):

- 64 transformer blocks, all with FFN. Hidden size `n_embd = 5120`,
  intermediate size `n_ff = 17408`. The arch is a hybrid dense Qwen3.5 variant
  (`attn_qkv`, `attn_gate`, `ssm_*` tensors are present).
- `blk.N.ffn_down.weight.in_sum2` is length 17408 for every layer, confirming it
  is the per-intermediate-channel salience vector we want.
- `17408 = 68 * 256`. With k-quant block alignment and uniform experts, the only
  evenly divisible expert counts are `E in {2, 4, 17, 34, 68}`. `E = 17`
  (1024 channels per expert) or `E = 34` (512) are the usable granularities.
- Salience is flat: on average the top 25% of channels carry only 59% of the
  activation mass, and about 71% of channels are needed for 90%. Static
  importance pruning alone will therefore degrade a lot. The dynamic per-token
  routing has to provide the win, and if the routing spread is as flat as the
  aggregate, the idea may not hold. This is the main Phase 0 question.

Note that the imatrix is an aggregate over the calibration set. It does not tell
us the per-token routing spread, which is what actually matters. Phase 0 must
measure that directly.

## Proposed approach

Per FFN layer, with `n_ff` intermediate channels:

1. Compute a salience `s[i]` per channel from `ffn_down.in_sum2[i]`, optionally
   multiplied by the L2 norm of `W_down[:, i]` (first-order contribution proxy).
2. Partition the `n_ff` channels into `E` pseudo-experts. Boundaries must align
   to the quantization block size so that gate/up row-slices and down
   column-slices stay block-aligned. On the test target that means `E in
   {17, 34}` for uniform experts.
3. Route each token to top-k pseudo-experts.
4. Compute only the selected experts; stream the unselected slices from the
   GGUF with the existing `madvise` machinery, reusing `prefetch_experts`,
   `maybe_pin_hot_experts`, and `evict_to_budget`.

Because gate/up slice along `ne1` and down slices along `ne0`, all three can be
exposed as expert-stacked ggml views with no data copy, if block alignment holds.
If a backend rejects non-contiguous expert tensors, fall back to a one-time
repack at load.

### Router candidates (the open question)

- R1, sketch router: for each expert, keep a small rank-r projection of its gate
  block, computed offline from `W_gate` by SVD. Score by `||P_e x||`. No
  training, cheap. Reuses the model's own gate geometry.
- R2, distilled router: run the frozen model over calibration text, record the
  oracle top-k experts per token using the true channel contributions, then fit a
  small linear router `[n_embd, E]` on CPU. This tunes a router, not the model.
- R3, static control: no per-token routing. Always activate the globally most
  important channels and stream the tail without dropping it. This is the
  baseline that isolates the value of dynamic routing.

R3 is the control. R1 and R2 are the candidates. Phase 0 picks between them
before any llama.cpp graph work.

## Phased plan

### Phase 0: offline feasibility (no C++ changes)

Goal: prove whether a training-free partition plus router can keep quality.

1. Add an offline analysis script that reads the GGUF metadata and the imatrix
   and prints per-layer `n_ff`, salience distribution, and candidate partitions.
2. Build the oracle top-k selection per token from the true gate/up/down
   contributions on a small calibration set.
3. Compare R1, R2, R3 against the full dense FFN on reconstruction error and a
   perplexity proxy.
4. Gate: if no scheme reaches the target quality at the target active fraction,
   stop and report. Do not start the C++ work.

### Phase 1: loader support (root Guanaco, minimal)

1. New registration path for dense FFN pseudo-experts, bypassing the
   `num_experts <= 1` disable only when an imatrix is present and the user asked
   for the mode. Keep the current dense behavior when the mode is off.
2. Extend the imatrix loader use to read per-channel `sums` for the dense path.
3. A new flag, name to be decided (`--guanaco-dense-*`), plus env override, in
   the existing `common-arg.cpp` / `common-common.*` patches.

### Phase 2: graph routing (llama.cpp patch)

1. Inject a top-k pseudo-expert path into the dense FFN build, reusing
   `build_moe_ffn` if the sliced views are accepted, otherwise a small sibling
   builder.
2. Emit a `ffn_moe_topk-<il>` tensor so the existing eval callback fires without
   changes.
3. Keep the patch minimal and per-arch first; a second arch only after the first
   works.

### Phase 3: end-to-end streaming run

Run the Q4_K_M model with the mode on and `-ngl 0 --load-mode streaming`, measure
RAM and speed, and compare completions to the full run.

## Metrics

- Quality: perplexity and a small zero-shot set versus the full dense run.
- Memory: peak RSS and resident expert fraction at run time.
- Speed: tokens/s versus full mmap and versus a reduced-context full run.
- Streaming health: resident percent and churn percent from the existing
  `[Guanaco HerdCache]` logs.

## Success criteria

- A measurable RAM reduction against the full dense run.
- Quality loss no worse than a same-size lower quant, and ideally small.
- No crash, no GPU VM error, coherent output on a fixed prompt set.

## Touchpoints

- `src/guanaco_steppeloader.cpp`: dense registration, per-channel salience,
  expert partition.
- `src/imatrix-loader.*`: per-channel `sums` access for dense tensors.
- `patches/llama-model.cpp.patch`: dense pseudo-expert registration.
- `patches/llama-graph.cpp.patch` (new): top-k routing for the dense FFN.
- `patches/llama-context.cpp.patch`: unchanged if we keep the `ffn_moe_topk` name.
- `patches/common-arg.cpp.patch`, `patches/common-common.cpp.patch`: new flag.

One patch per upstream file. Regenerate patches after changes and verify by
applying them to a pristine checkout.

## Risks

- Quant block alignment may forbid clean expert boundaries. Mitigation: choose E
  so `n_ff / E` is a multiple of the block size, or use uneven expert sizes.
- Salience is flat on the test target (top 25% of channels carries only 59% of
  mass). If per-token routing is equally flat, there is little to route and the
  idea fails. Mitigation: Phase 0 oracle measurement of the per-token spread
  before any C++ work.
- Non-contiguous expert views may be rejected by `mul_mat_id` backends.
  Mitigation: one-time repack at load.
- A training-free router may lose too much quality. Mitigation: Phase 0 gate.
- Dense split touches a widely used code path. Mitigation: keep it behind the
  new flag and off by default.

## Verification

- Root `cmake -S . -B build` applies the full patch set cleanly.
- Root build passes.
- Patches reproduce from pristine `llama.cpp` at the pinned submodule commit.
- Streaming run on the test target shows the expected logs and no crash.

## Phase 0 results (routing spread)

Status: measured. Harness is `examples/tomoe_probe.cpp` (build with
`-DGUANACO_BUILD_PROBES=ON`). It runs the dense Q4_K_M model on CPU, captures
`ffn_swiglu-<il>` per token, buckets channels into E uniform experts, and reports
the share of per-token activation energy captured by a given kept subset.

Run: 736 tokens of varied prose, all 64 layers, `-ngl 0`.

- The capturable win is the oracle top-k (each token picks its own experts).
  This is an upper bound: no training-free router can beat it.
- Static top-k is the aggregate/global ranking, the control (what imatrix
  ranking would keep).

Capture of per-token SwiGLU energy, averaged over all layers:

| kept % | E=17 oracle | E=17 static | E=34 oracle | E=34 static |
|--------|-------------|-------------|-------------|-------------|
| 23.5%  | 0.401       | 0.279       | 0.440       | 0.286       |
| 50%    | 0.660       | 0.555       | 0.681       | 0.562       |
| 70%    | 0.814       | 0.757       | 0.831       | 0.759       |
| 88%    | 0.933       | 0.907       | 0.940       | 0.910       |
| 100%   | 1.000       | 1.000       | 1.000       | 1.000       |

Per-token spread is real but weak: effective expert count is about 14.5 of 17,
and even the oracle needs roughly 85% of channels kept to retain 90% of energy.
Dynamic routing beats static by 8 to 15 points across the range, but the absolute
retention at aggressive compression is low. A few early layers (0 and 2) are more
concentrated; most middle layers are nearly uniform.

Caveat: this is unweighted activation energy. The true FFN output is
`ffn_down @ swiglu`, so low-energy channels with small down-projection columns
may cost less than the energy share suggests. The proxy does not include that
weighting or cancellation, so it is a first-order indicator, not ground truth.
The next decisive step is a real quality measurement (perplexity under an actual
drop), not more proxy work.

## Pivot: quality-preserving dense layer-window streaming

Decision: shelve the pseudo-expert idea. Stream whole dense weight tensors from
disk instead, with no weight changes and no quality loss. Reuse the SteppeLoader
pager.

### Mechanism

- Register every large dense weight tensor (FFN gate/up/down, attn qkv/gate/out,
  ssm tensors) as a streamable unit with `(layer, file_offset, byte_size)`. The
  unit is the whole tensor, so `num_experts == 1`. The existing per-slice
  machinery is bypassed; the pager works on whole-tensor page ranges.
- Sliding layer window: keep layers `[cur - W + 1, cur]` resident. On each layer
  boundary, `MADV_WILLNEED` the next `ahead` layers and `MADV_DONTNEED` layers
  that fell behind the window. `W` is derived from a new byte budget.
- New config `dense_budget_mb`. Window `W = budget / mean_layer_bytes`.
- Trigger: a new hook method `on_layer_boundary(il)`, called from the existing
  eval callback on a dense per-layer node (`l_out-<il>` or `ffn_out-<il>`). Small
  addition to `patches/llama-context.cpp.patch`. The MoE path is untouched.
- imatrix: only sets pin priority when the budget cannot hold the whole model.
  Honest note: for a dense model every tensor is used every token, so pinning
  does not change per-token I/O. It only chooses which bytes stay cached. imatrix
  cannot reduce streamed bytes for dense the way it can for sparse MoE routing.

### Honest expectations

Dense streaming reads `(1 - resident_fraction) * model_bytes` from disk every
token. The 27B Q4_K_M is 16.6 GB; at a 10 GB budget that is about 6.6 GB per
token. On NVMe that is roughly 1 to 2 s/token. This is only useful when RAM is a
few GB short of the model, or for offline/batch work, not interactive chat. It is
not a way to run a much larger model on a small box.

For a model that fully fits, plain mmap plus the OS page cache already does the
same thing, so the feature only earns its place when it beats the OS under a
memory cap. That must be measured first.

### Milestones

- D1 (no code): baseline. Run the 27B under `MemoryMax` caps (e.g. 12G, 10G, 8G)
  with the OS page cache doing the work, and record tokens/s and peak RSS. This
  sets the bar the explicit pager must beat.
- D2: dense registration plus the sliding-window pager in `SteppeLoader`, and a
  `dense_budget_mb` flag in the existing CLI patches.
- D3: layer-boundary hook and an end-to-end run; compare against D1.

If D1 shows the OS already matches what the pager can do, stop before D2.

### D1 baseline results

Two environment findings that shape the whole test:

- `-ngl 0` does not keep the model on CPU here. The box has an AMD Radeon
  840M iGPU (UMA, 24 GB share). By default llama.cpp offloads the whole model
  to it (weights live in `/dev/dri/renderD128` Vulkan buffers, process RSS stays
  tiny). To get a file-backed CPU run you need `-dev none`.
- On CPU, weight repacking doubles memory (26 GB RSS for a 16.6 GB model).
  `-nr/--no-repack` gives a file-backed 16.1 GB RSS and is faster. Streaming is
  only meaningful with `-nr`, because only then are the weights the mmap pages
  that `madvise` controls.

Baseline (CPU, `-dev none -nr`, generation-style `-b 1 -ub 1`, 16 tokens,
model evicted from page cache first, cgroup `MemoryMax`, `MemorySwapMax=0`):

| MemoryMax | peak RSS | tok/s | s/token |
|-----------|----------|-------|---------|
| none      | 16.1 GB  | 2.237 | 0.45    |
| 12 G      | 12.3 GB  | 0.223 | 4.5     |
| 10 G      | 10.3 GB  | 0.197 | 5.1     |
| 8 G       | 8.2 GB   | 0.022 | 45      |

Under a cap the OS page cache delivers roughly 1 GB/s of effective weight
throughput and collapses to thrash at 8 G. The theoretical floor at a 12 G cap
is about 5 GB/token divided by NVMe bandwidth (several GB/s), so there is
several times of headroom for explicit `MADV_WILLNEED` prefetch to recover. That
is the specific gap D2 and D3 target.

Note: the iGPU can hold the whole 16.6 GB model and ran a 256-token prefill at
about 28 tok/s. On this exact machine the model fits both in RAM and in the
iGPU, so dense streaming is only interesting as a constrained-budget experiment.

## D2/D3 implementation and results

Status: implemented and measured. All changes are in-tree on `feat/tomoe`; the
llama.cpp edits are in `patches/*.patch` (regenerated and verified to apply to a
pristine submodule).

### What D2/D3 do

The first design was a sliding layer window `[cur-W+1, cur]`. That cannot reduce
disk traffic: a layer evicted at the start of a pass is needed again only at the
end of the next pass, so every pass re-reads the full streamed tail anyway. The
window only fires `MADV_WILLNEED` / `MADV_DONTNEED`, which on the shared file mmap
is a no-op for DONTNEED, and the cgroup reclaim raids the window. Net effect:
about one full model read per token, and no speedup over the OS.

The shipped design instead keeps a FIXED resident subset and streams the tail:

1. Register every transformer-block weight (skip `norm` tensors) with
   `(layer, file_offset, byte_size)`. Dense registration lives in
   `patches/llama-model.cpp.patch`.
2. At load, allocate a private anonymous slab per streamed tensor and redirect
   `ggml_tensor::data` to the slab (`get_stream_tensor_data`,
   `src/guanaco_model_hook.cpp`). Weights are whole-tensor copies; no weight is
   ever edited, so quality is unchanged.
3. Fill the fixed resident subset `[0, W)` once (`fill_resident_layers`), where
   `W = dense_budget_bytes / mean_layer_bytes`. Those layers are never evicted
   and never re-read.
4. On each layer boundary (`l_out-<il>` callback, `patches/llama-context.cpp.patch`)
   prefetch the next streamed layers `ahead` early via a batched io_uring read
   into their slabs, then evict the layer just computed. Only the streamed tail
   `[W, n)` is paged per token.

`fadvise(POSIX_FADV_DONTNEED)` is issued after each read so the kernel page cache
copy is dropped; without it the tail crowd out the cgroup and the run slows down.

### The cap floor is anonymous memory, not page cache

Sampling `/proc/<pid>/smaps_rollup` and the cgroup `memory.stat` at steady state
(budget 8192, cap 14G) showed the cgroup is 13.0 GB anon + 1.0 GB file. So the
file cache is NOT the floor; the float is our own anonymous slabs. Breakdown:
8.4 GB private-dirty (resident slabs + ggml buffers) plus about 5.4 GB of
private-clean / lazyfree pages.

That 5.4 GB was the bug. `MADV_FREE` only marks pages; the kernel never bothers
reclaiming them while `memory.current` sits just under `memory.max`, so evicted
streamed slabs keep counting against the cap and every allocation pays
synchronous direct reclaim. The fix is to free eagerly under pressure.

### Eviction advice

`evict_layer` chooses the madvise advice per eviction (env override
`GUANACO_DENSE_EVICT=free|dontneed`, default adaptive):

- `MADV_FREE`: cheap refill (no zero-fill), pages stay resident as cache. Wins
  when the cap has headroom.
- `MADV_DONTNEED`: frees immediately, keeps the cgroup under the cap, but the
  refill pays a zero-fill (visible as tens of millions of minor faults). Wins
  when the cap is tight.
- adaptive (default): `MADV_FREE` while `memory.current < 90% of memory.max`,
  else `MADV_DONTNEED`.

Same harness (budget 8192, ahead 4, 16 gen tokens, prompt "The capital of France
is", model cache evicted first, cgroup `MemoryMax` + `MemorySwapMax=0`):

| cap | free | dontneed | adaptive |
|-----|------|----------|----------|
| 12G | 0.2  | 0.4      | 0.4      |
| 14G | 0.3  | 0.4      | 0.4      |
| 16G | 0.5  | 0.5      | 0.5      |
| 18G | 0.6  | 0.4      | 0.5      |

Adaptive is the best all-round choice: it matches DONTNEED at tight caps and
keeps most of FREE's caching with headroom.

### Adaptive results, resident budget vs cap

A larger resident budget at a given cap is always better: resident layers are
never re-read. Adaptive eviction, ahead 4:

| cap | budget MiB | resident | tok/s | elapsed | peak RSS |
|-----|-----------|----------|-------|---------|----------|
| 6G  | 4096      | 18       | 0.2   | 2:48    | 6.0 GB   |
| 8G  | 4096      | 18       | 0.2   | 2:30    | 8.0 GB   |
| 10G | 8192      | 37       | 0.3   | 1:46    | 10.0 GB  |
| 12G | 8192      | 37       | 0.4   | 1:36    | 12.6 GB  |
| 14G | 8192      | 37       | 0.4   | 1:27    | 14.7 GB  |
| 16G | 8192      | 37       | 0.5   | 1:20    | 16.8 GB  |
| 18G | 8192      | 37       | 0.5   | 1:11    | 18.9 GB  |

Peak RSS tracks the cap, so the pager genuinely respects the budget down to 6G
(about 2.8x below model size).

Comparison points at cap 12G: the recorded D1 OS baseline is 0.223 tok/s, the
pager gives 0.4 (about 1.8x). A cold-cache pager-off run under this exact
harness did not finish generation sanely for 16 tokens (16:47 wall, 639 s system
time, 77.8M major faults), so against a truly cold OS page cache the gap is much
larger.

### What is left

- I/O still runs synchronously inside the layer-boundary callback, so it does not
  overlap compute. At cap 12G the summed slab-read time is about 68 s of a 96 s
  run. Submitting the ahead reads without reaping and only gating before the
  layer is computed should hide part of that. This is the largest remaining
  lever.
- The refill zero-fill (54M minor faults at cap 12G) is inherent to eager
  freeing; transparent huge pages on the slabs might cut the fault count.


