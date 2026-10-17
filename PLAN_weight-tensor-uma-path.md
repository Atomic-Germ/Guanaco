# PLAN: Weight-Tensor UMA Path

## Goal

Enable a direct GPU compute path for weight tensors on UMA systems
(AMD Zen4/Zen5 APUs with integrated GPU) that avoids all PCIe bus copies.
The GPU reads weights from host-visible UMA memory directly — no staging
buffer, no device-local copy, no round-trip.

## Background

The existing `gpu_pill` feature creates a UMA buffer type for KV cache
writeback. The scheduler patch (`ggml-src-ggml-backend.cpp.patch`) already
avoids cross-backend copies when a tensor's buffer is host-resident
(`ggml_backend_buft_is_host`). However, the weight path is deliberately
pinned to CPU (`n_gpu_layers=0` forces `n_gpu_layers=0`), so weights never
reach a UMA buffer — they stay in the file mmap and are read by the CPU.

On a UMA system the RAM is physically one pool, so "host-visible" memory
is GPU-accessible memory. The building blocks are:

- **UMA buffer type** (`ggml_backend_vk_buffer_type_uma`): allocates
  `eHostVisible | eHostCoherent | eHostCached` memory, registers in
  `pinned_memory` for `ggml_vk_host_get()` offset resolution.
- **Scheduler sync-backend mechanism**: when a tensor's buffer is
  host-resident, the scheduler waits on the source backend's event instead
  of creating a copy.
- **Streaming engine** (`guanaco_steppeloader.cpp`): in-place mmap paging
  via `madvise`, slab redirection, io_uring reads.

The missing piece: a way to place **weight tensors** into UMA host-visible
memory so the GPU can read them directly during prefill and decode.

## Design

### Overview

When `gpu_pill` is active (UMA device detected, no discrete GPU):

1. After model loading, allocate a UMA buffer for each weight tensor.
2. Copy weights from the file mmap into the UMA buffer.
3. Point the ggml tensor's `->data` at the UMA buffer.
4. The scheduler's existing sync-backend mechanism sees the UMA buffer is
   host-resident, so it waits on the CPU backend's event instead of creating
   a device-local copy.

This gives: **disk → mmap → memcpy → UMA host-visible → GPU reads directly**.
One memcpy at load time, zero PCIe copies during inference.

### Compatibility Gating

The path is gated by the existing `gpu_pill` detection:

```
gpu_pill_allowed = has_igpu && !has_gpu && !has_meta
```

No new detection logic is needed. When `gpu_pill` is disabled, the system
falls back to the existing CPU-offload path (current behavior).

### Interaction with Streaming

When both `--load-mode streaming` and `--gpu-pill` are active:

- **MoE experts**: The streaming engine already pages experts in/out via
  `madvise` on the file mmap. With UMA, the paging target is the UMA buffer
  instead of the file mmap. The router prefetch (`on_router_computed`)
  preads expert slices into the UMA slab.
- **Dense layers**: The layer-window streaming pager writes into UMA slabs
  instead of anonymous memory. The pager's `t->data` redirection targets
  the UMA buffer.
- When `n_gpu_layers=0` is forced (streaming mode), the UMA path overrides
  this — weights go to UMA memory, not device-local memory. The GPU reads
  them directly.

### Allocation Strategy

Two modes:

| Mode | Behavior | Use case |
|------|----------|----------|
| **Eager UMA** | All weight tensors allocated to UMA at load time | Models that fit in RAM |
| **Streaming UMA** | Only resident expert slabs use UMA; cold experts stay on disk | Models larger than RAM |

Eager UMA is simpler and subsumes the current `gpu_pill` KV-only behavior
if desired. Streaming UMA is the combination of the two features.

## Implementation Steps

### Step 1: UMA buffer for weight tensors in `llama_model_base::load_tensors`

**File**: `patches/llama-model.cpp.patch`

After the existing Guanaco streaming tensor-redirection block, add a new
block that iterates all weight tensors (not just experts) and allocates
UMA buffers for them:

```cpp
#ifdef GUANACO_ENABLED
if (gpu_pill_active) {
    for (auto & [ctx, buf_map] : ctx_buf_maps) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx);
             t != nullptr;
             t = ggml_get_next_tensor(ctx, t)) {
            // Skip tensors already on a UMA buffer
            if (t->buffer && t->buffer->buft->iface.is_host &&
                t->buffer->buft->iface.is_host(t->buffer->buft)) {
                continue;
            }
            // Skip non-host tensors (already on GPU)
            if (t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
                continue;
            }
            // Allocate UMA buffer and copy weights
            auto * uma_buft = /* get UMA buft for device */;
            size_t nb = ggml_nbytes(t);
            ggml_backend_buffer_t uma_buf =
                ggml_backend_buft_alloc_buffer(uma_buft, nb);
            void * dst = ggml_backend_buffer_get_base(uma_buf);
            memcpy(dst, t->data, nb);
            ggml_backend_buffer_free(t->buffer);
            t->buffer = uma_buf;
            t->data = dst;
        }
    }
}
#endif
```

**Key considerations**:
- Only redirect tensors that are currently host-resident (file mmap or
  CPU buffer). Skip tensors already on a GPU device buffer.
- Use `ggml_backend_dev_get_uma_buft()` (already implemented in the
  kv-cache patch) to get the UMA buffer type.
- The copy is a one-time cost at load time.

### Step 2: Thread `gpu_pill_active` through model loading

**Files**: `patches/llama-model.cpp.patch`, `patches/llama-context.cpp.patch`

The context already computes `cparams.gpu_pill`. Pass it to the model's
`load_tensors` or store it on `llama_model_base` so the UMA weight
allocation block can check it.

Add to `llama_model_base`:
```cpp
bool gpu_pill_active = false;  // set from context params
```

### Step 3: Scheduler integration (already implemented)

No new scheduler work needed. The existing sync-backend mechanism in
`ggml-src-ggml-backend.cpp.patch` already handles host-resident buffers:

```cpp
} else if (src->buffer && ggml_backend_buft_is_host(src->buffer->buft)) {
    ggml_backend_sched_split_sync_add(split, src_backend_id);
}
```

When the UMA buffer is host-resident, the scheduler records the CPU
backend as a sync backend and waits on its event instead of creating a
copy. The GPU reads directly from the UMA buffer.

### Step 4: MoE expert streaming into UMA slabs

**Files**: `patches/llama-model.cpp.patch`, `src/guanaco_model_hook.cpp`, `src/guanaco_steppeloader.cpp`

When `gpu_pill && streaming`:

1. `register_expert_tensor()` allocates the slab in UMA memory instead of
   anonymous memory.
2. `on_router_computed()` preads expert slices into the UMA slab.
3. The ggml tensor's `->data` points at the UMA slab (already handled by
   the existing `t->data = slab` redirection in dense mode; extend to
   expert mode).

In `guanaco_model_hook.cpp`, the slab allocation changes from:
```cpp
void * slab = malloc(nb);  // or mmap anonymous
```
to:
```cpp
ggml_backend_buffer_t uma_buf =
    ggml_backend_buft_alloc_buffer(uma_buft, nb);
void * slab = ggml_backend_buffer_get_base(uma_buf);
```

### Step 5: Dense layer-window streaming into UMA slabs

**Files**: `patches/llama-model.cpp.patch`, `src/guanaco_steppeloader.cpp`

Same principle as Step 4 but for the dense pager. The layer-window
streaming allocates private slabs via `get_stream_tensor_data()`. Change
the slab allocation to use UMA buffers when `gpu_pill` is active.

### Step 6: CLI and parameter plumbing

**Files**: `patches/common-arg.cpp.patch`, `patches/common-common.h.patch`

The `--gpu-pill` flag already exists. No new CLI flag needed — the weight
UMA path is an implementation detail of `gpu_pill` on UMA systems.

Optionally add `--gpu-pill-weights` as a separate control for users who
want KV-only UMA (current behavior) vs full weight UMA (new behavior).

### Step 7: Op offload tuning

**File**: `patches/llama-context.cpp.patch`

When `gpu_pill` is active and weights are in UMA memory, lower the
`GGML_OP_OP_OFFLOAD_MIN_BATCH` to 1 so all ops go to the GPU:

```cpp
if (cparams.gpu_pill && getenv("GGML_OP_OFFLOAD_MIN_BATCH") == nullptr) {
    llama_backend_set_op_offload_min_batch(backends, 1);  // was 2
}
```

This is a tuning knob — batch=1 means even batch-1 prefill ops go to the
GPU, which is beneficial on UMA systems where the GPU can read weights
directly.

## Testing

### Unit-level

- Verify UMA buffer allocation succeeds on a UMA device.
- Verify `memcpy` from mmap to UMA produces identical inference output.
- Verify scheduler event-wait path fires for UMA-backed tensors (no copy
  created).

### Integration

- Run `llama-bench` with `--gpu-pill` on a UMA system and confirm:
  - No device-local weight buffer allocations (check with
    `GGML_VK_DEBUG=1` or memory stats).
  - GPU compute reads from host-visible addresses.
  - Inference output matches the CPU-offload baseline.
- Run with `--load-mode streaming --gpu-pill` and confirm:
  - Expert preads target UMA slabs.
  - No PCIe copies during decode (measure with `perf` or GPU perf counters).

### Correctness

- Compare output tokens between:
  - CPU-only baseline (`-ngl 0`)
  - Current `gpu_pill` (KV-only UMA)
  - New `gpu-pill` (weight UMA)
- All three should produce identical output for the same prompt and seed.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| UMA buffer allocation fails (OOM on small-RAM APUs) | Fall back to CPU path with a warning. Check `uma_buft` return value. |
| `memcpy` at load time is slow for large models | Show progress indicator. This is a one-time cost. |
| Scheduler sync-backend wait causes stalls | The CPU backend event should fire quickly since it's just a fence. Verify with perf. |
| Interaction with `mmap` + `madvise` is complex | When UMA is active, skip `MADV_DONTNEED` on the original mmap region that's been copied to UMA. |
| Non-UMA systems accidentally trigger the path | The `gpu_pill_allowed` gate (`has_igpu && !has_gpu && !has_meta`) is strict. |
| MXFP4 / quant formats with block scales | Skip UMA redirection for formats where flat memcpy is unsafe (already handled in streaming). |

## Design Decisions

### D1: Weight UMA is the default for `gpu_pill`, with fallback

Weight UMA is a **silent upgrade** — if someone runs with `--gpu-pill` on
a UMA system (which is extremely likely if not universal), they get the
weight UMA path automatically. No new CLI flag. If UMA buffer allocation
fails (OOM, unsupported device), fall back to the existing CPU-offload path
with a warning.

### D2: `n_gpu_layers=0` is harmless, leave as-is

The existing `n_gpu_layers=0` force for streaming/gpu-pill is not a
problem on UMA systems — it's a no-op since the UMA buffer is host-visible
regardless. Users who set it themselves will continue to do so. No change
needed.

### D3: Dense-only is acceptable; dense + MoE is ideal

The weight UMA path should work for both dense and MoE models. Since
`--gpu-pill` became a thought, more dense-related work has been done, so
the dense path is the primary target. The MoE path (expert slabs in UMA)
is the ideal end state but can follow the dense implementation.

**MoE research needed**:
- How `moe_cache_size` interacts with UMA slab allocation
- Whether the router prefetch (`on_router_computed`) can target UMA slabs
  without changes or needs explicit UMA-aware pread
- Whether the two-pass eviction (PASS A/B) in the streaming engine needs
  UMA-specific handling
- Whether `register_expert_tensor` slab allocation can be made UMA-aware
  transparently

### D4: Backward compatibility across AMD APUs

llama.cpp itself has no explicit Zen5 code path, and other AMD APUs exist
(gfx1032, gfx1103, gfx1150, etc.). The UMA path must:
- Work on any Vulkan-capable AMD iGPU, not just gfx1152
- Not assume AVX-512 (Zen4 has AVX-512 too, but older APUs may not)
- Detect UMA capability via `GGML_BACKEND_DEVICE_TYPE_IGPU` (already done)
- Fall back gracefully on non-AMD iGPUs (Intel, ARM)
