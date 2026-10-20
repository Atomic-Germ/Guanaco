# The llama-server / Vulkan KV crash: what is actually known

A running record, because this has been mis-attributed three times and the
corrections matter more than the conclusion.

## Symptom

`llama-server` with a GPU-resident KV cache dies partway into a multi-request
session. Two observed sites, both moving KV data:

```
(A) get_available_slot -> prompt_save -> state_seq_get_data
      -> llama_io_write_host -> ggml_backend_vk_buffer_get_tensor
      -> ggml_vk_buffer_read_2d              <- SIGSEGV in memmove

(B) get_available_slot -> selected slot by LCP similarity
      -> prompt processing / KV shift        <- dies later in the response
```

Reproduced 4/4 on (A) with an identical stack.

## What was wrong with earlier conclusions

1. **"It is upstream, not our UMA allocation."** The evidence for this was that
   it also faulted with gpu-pill *off*, i.e. with a plain Vulkan buffer. That
   part was right. But the same reasoning was then used to claim it was *not a
   Vulkan problem* - and it was never tested against unpatched upstream.

2. **"It is the Vulkan stack."** The Lunar SDK rebuild removed the load-time
   crashes and the 111B OOM, so the stack was genuinely broken. But the KV
   crash survived with a byte-identical stack. Two separate problems, and the
   fix for one did nothing for the other.

3. **Isolation runs that proved nothing.** The first flag isolation inherited a
   shell full of `LLAMA_ARG_*` - including `LOAD_MODE=streaming` - so every case
   ran with the working configuration and all six "survived", including the case
   meant to be the crashing default. The second isolation lost `-ngl` to
   `env`'s argument parsing and every case ran with a bare `99`. Both produced
   clean-looking tables that measured nothing.

## Established

- **The discriminator is `LOAD_MODE`.** Plain mmap (or AUTO) crashes on request
  2. `LOAD_MODE=streaming` survives 5 requests. Streaming keeps the KV in host
  memory, so it never enters the crashing path.
- **Not architecture-specific.** Both the 9B dense and the 24B MoE crash
  identically.
- **The KV cache location follows `offload_kqv`, not `n_gpu_layers`.** This is
  why `-ngl 99` fails while gpu-pill works: streaming forces `n_gpu_layers=0`
  and keeps the KV in host memory, whereas `-ngl 99` leaves KV offload enabled
  and the KV lands in a Vulkan buffer.
- **A likely mechanism, from the server's own warnings:**

  ```
  W llama_context: n_ctx_seq (16128) > n_ctx_train (8192) -- possible training context overflow
  W srv load_model: the slot context (16128) exceeds the training context of the model (8192) - capping
  ```

  The requested context is capped down to the training context, but if an
  allocation is sized from the *requested* value while the tensors are sized
  from the *capped* one, a host-side read runs past its destination. That is
  exactly the shape of a memmove fault, and it would be independent of Vulkan
  in origin - Vulkan merely being where the read happens.

## Mitigation

**Do not request a context larger than the model was trained for.** `ctx=0`
lets llama.cpp use the model's own `n_ctx_train`, so no capping occurs and the
mismatch cannot arise. This is the rule to document, and it is free.

## Resolution

**It is ours, and it is fixed.**

`attrib_final.sh` built an unpatched llama.cpp at the pinned commit `aa94f2086`
with Vulkan enabled and ran the identical scenario:

| | result |
|---|---|
| ours, `-ngl 99` | **CRASHED on request 2** |
| ours, `-ngl 99 --no-kv-offload` | survived 5 requests |
| vanilla, `-ngl 99` | survived 5 requests |
| vanilla, `-ngl 99 --no-kv-offload` | survived 5 requests |
| vanilla, `-ngl 0` | survived 5 requests |

So the earlier conclusion that this was an upstream bug was wrong, and the
vanilla build is what proved it. The trigger is a GPU-resident KV cache.

**The fix: gpu-pill keeps the KV cache in host memory.** Under gpu-pill the GPU
reads weights directly out of system RAM over UMA, so a separate GPU-resident KV
buffer cannot be faster than the host memory the weights are already streaming
from. It only introduces a second allocation discipline for a path that
llama-server's slot handling then has to move around.

`llama-context.cpp` now sets `cparams.offload_kqv = false` when `gpu_pill` is
active. Measured free - 9B dense, five requests each:

| configuration | tok/s |
|---|---|
| `-ngl 0` (CPU compute) | 3.52-3.60 |
| `-ngl 99` + gpu-pill (GPU compute, KV in host) | 3.35-3.42 |
| `-ngl 99` + gpu-pill + streaming | 3.53-3.54 |
| `-ngl 99` + streaming, 2 GB budget (forced paging) | 2.57-2.94 |

The GPU is not faster than six CPU threads for this model, so gpu-pill's value
is that it leaves host memory free for weights, not that it accelerates
compute.

## A second, separate Vulkan problem

The locally built Lunar SDK removed the load-time crashes and the OOM on the
111B model, which had been failing for the same reason as the KV crash. Those
were genuine driver-stack faults and are fixed. They are unrelated to the KV
crash above, which survived the SDK rebuild with a byte-identical stack.

## Methodology note

Every isolation run must control its environment explicitly - `env -i` plus
flags placed *after* the binary - and must include a case that is expected to
crash, to prove the harness can detect a crash at all. Three conclusions in
this project were wrong for want of exactly that.
