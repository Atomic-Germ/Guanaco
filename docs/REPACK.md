# Per-expert repack and disk streaming: why the combination is a dead end

A design note for anyone who later tries to combine llama.cpp's weight-repacking
optimization with Guanaco's expert disk streaming. Short version: the data
layouts are compatible (verified byte-for-byte), but the architectures cannot
share the same memory, and the only alternative reintroduces the design Guanaco
already abandoned.

## The goal

Stock llama.cpp (with `use_extra_bufts` enabled, the default) allocates each
quantized tensor into a `CPU_REPACK` buffer at load time. `set_tensor` then
transforms the weight into an interleaved layout (for example `iq4_nl_8x8` on
AVX2) that feeds the int8 vectorized `gemv`/`gemm` kernels in
`ggml/src/ggml-cpu/repack.cpp`.

Guanaco streaming mode disables this entirely: `use_extra_bufts = false`
(patches/common-common.cpp.patch, `--no-repack` implied). The patch comment
says why: the CPU_REPACK pass copies every weight into RAM, defeating disk
streaming.

The consequence is a small output divergence between streaming and default
mode. The repacked int8 dot-product path accumulates differently than the
direct dequantize-and-accumulate path, so logits differ slightly and near-tie
tokens can flip. An A/B of the same model with repack on vs off (both on
mmap, same inputs) flipped one token in 32.

The question investigated here: can the active experts be repacked per expert,
per layer, into a small bounded slab, so streaming keeps RAM bounded but
produces the exact same numerics as default mode?

## Why streaming disables repack today

Repack is driven entirely by the tensor's buffer type. In
`ggml/src/ggml-cpu/repack.cpp`:

- `extra_buffer_type::supports_op()` only claims a `MUL_MAT`/`MUL_MAT_ID` op
  when `op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type()`.
- `extra_buffer_type::get_tensor_traits()` returns traits under the same gate,
  and the CPU backend consults them in `ggml_cpu_extra_compute_forward()`
  (traits.cpp) before running the normal compute.
- The repacked bytes are produced exactly once, in
  `ggml_backend_cpu_repack_buffer_set_tensor()`, during model load.

A streamed expert tensor lives in the regular file-backed CPU mmap buft, so the
repack path never fires for it. Every layer's `mul_mat_id` uses the direct
path, reading the original quantized layout straight from the mmap.

## The good news: per-expert repack is byte-identical to fused repack

The layout question is the one thing that could have made the whole idea
impossible before it even started. It turns out to be fine.

The repack driver (`repack_iq4_nl_to_iq4_nl_8_bl`, same file) processes rows in
groups of 8 and emits self-contained 8-row blocks:

```
for (b = 0; b < nrow; b += 8)
    for (x = 0; x < nblocks; x++)
        for (i = 0; i < 8; i++) dst_tmp[i] = src[x + i*nblocks]
        *dst++ = make_block_iq4_nlx8(dst_tmp, 8)
```

An expert tensor is 3D `{in, out, experts}` and the traits gate requires
`ne[1] % 8 == 0`, where `ne[1]` is the per-expert `out` dimension. So every
8-row group lies entirely inside one expert, and the output offset of each
expert's repacked bytes is exactly its offset in the raw tensor.

We verified this empirically: repack a fused `{64, 32, 4}` IQ4_NL tensor and
an isolated `{64, 32}` expert slice from the same raw data, then compare the
expert's bytes in the fused result against the isolated result.

```
repack: repack tensor with iq4_nl_8x8   (both picked the same traits)
MATCH: per-expert repack is byte-identical to fused repack slice
```

So the resident experts could be repacked one at a time into a slab at their
fused offsets and the repack `mul_mat_id` kernel would read them exactly as it
reads a fully repacked tensor. Condition: per-expert `out % 8 == 0` for the
AVX2 `iq4_nl_8x8` traits (true for essentially every MoE layer in production;
the same reasoning applies to the other repack traits and their row-group
sizes). See the appendix for the check.

## Why it is still a dead end

### 1. The dispatch gate needs a new buffer type

To reach the repack kernels, a tensor must sit in a buffer whose type is the
repack buft. The streamed experts are in the mmap buft, and they cannot be
moved to the repack buft without copying the whole tensor into RAM (which is
the thing streaming exists to avoid). A custom "streaming repack" buft with its
own `extra_buffer_type` registered in `ggml_backend_cpu_get_extra_buffer_types()`
is possible in principle, but it is an invasive change to ggml core and it only
gets you to the second, more fundamental problem.

### 2. The madvise model has a hard invariant: the mapping is never written

Guanaco streams in place on the file-backed mmap (which llama.cpp creates with
`MAP_SHARED`):

- `evict_to_budget()` drops cold slices with `MADV_DONTNEED`
  (src/guanaco_steppeloader.cpp).
- A later access page-faults the same bytes back from the GGUF file.

This is safe only because the mapped expert bytes are identical on every fault.
That invariant requires the pages to stay clean: if repacked bytes were written
into the mapping, the pages become dirty, and then either

- `MADV_DONTNEED` on a `MAP_SHARED` file mapping flushes the dirty pages back to
  the GGUF file (corrupting the model on disk), or
- the next fault re-reads the original layout, silently invalidating the repack
  cache.

Either way, in-place repack on the streamed mapping is not just slow - it is
unsafe. Repack fundamentally needs writable memory separate from the mapping.

### 3. Per-expert repack on stream == the architecture iteration 2 abandoned

Writable memory means a separate slab per fused tensor, and that is exactly the
design Guanaco iteration 2 used before the in-place madvise model replaced it
(the "Iteration 2" header comment in src/guanaco_model_hook.cpp and the
`get_expert_tensor_data()` tensor-data-redirection API are its remains). Its
costs, and why it lost:

- Every layer needs the active experts copied and transformed from the mmap
  into the slab before that layer's `mul_mat_id` runs. That is a serial
  per-layer cost on the critical path.
- Memory traffic for expert weights roughly doubles: read mmap, write slab,
  read slab in the kernel.
- It loses the zero-copy overlap the current design gets for free: page faults
  and `MADV_WILLNEED`/PILOT prefetch happen behind compute, because the kernel
  reads the mmap directly.
- It re-introduces the ordering hazard with the load-time CPU_REPACK pass (the
  `on_tensors_loaded()` deferral in guanaco_model_hook.cpp).

The io_uring slab reader (src/guanaco_io_uring.cpp, `io_uring_slab_read`) was
written for this architecture. It is still in the tree, created and destroyed
but never called - the madvise path took over all I/O.

### 4. The payoff is marginal

The only thing per-expert repack-on-stream buys is numeric parity with default
mode. The observed divergence is a few near-tie token flips - the same class of
hardware-variant noise llama.cpp users already accept between instruction sets
and backends. It is not a correctness bug and it does not compound.

## Alternatives considered

### Pre-repack the GGUF so the mmap maps repacked bytes

The repacked 8x8 layouts preserve the original byte size for some types
(IQ4_NL, Q4_0, Q4_K, Q2_K) but not others (Q5_K, Q6_K), so offsets would shift
type-dependently. More fundamentally, the file would become Guanaco-specific:
the consumer would have to know the experts are pre-repacked, skip the load-time
transform, and dispatch the repack kernels from the mmap. Vanilla llama.cpp
would double-transform (garbage) or mis-dequantize it. Rejected.

### Whole-tensor repack into a slab at load

Equivalent to loading all expert weights into RAM, which is the exact cost
streaming removes. Rejected.

## For anyone revisiting this

If upstream ever integrates the streaming concept and wants repack numerics,
the realistic shape is: a custom buffer type for expert tensors whose memory is
a bounded writable slab, plus a `mul_mat_id` dispatch that reads per-expert
repacked data from the slab and falls back to direct-path streaming for
anything not resident. The per-expert layout proof above means the slab stores
repacked bytes at the fused tensor's natural per-expert offsets, and the gemv
kernels need no changes. The known costs are the serial copy+transform before
each layer and the lost mmap overlap; hiding it requires double-buffering plus
a repack variant of the PILOT lookahead prefetch.

The more pragmatic alternative, which this project chose, is to accept the tiny
logit difference and keep the zero-copy madvise model.

## Appendix: reproducing the byte-identity check

Link against the built `libggml-base`, `libggml`, and `libggml-cpu`; declare
`ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);`
(it is a C++ symbol). Create a 3D IQ4_NL tensor `{in, out, experts}`, a 2D
IQ4_NL tensor `{in, out}`, allocate both via
`ggml_backend_alloc_ctx_tensors_from_buft(ctx, repack_buft)`, fill both from the
same raw bytes (full tensor vs one expert's slice) with
`ggml_backend_tensor_set`, then memcmp the expert's byte range in
`fused->data` against `expert->data`. `out` must be divisible by 8. The two
buffers print `iq4_nl_8x8` in the repack debug log and the ranges are
byte-identical.
