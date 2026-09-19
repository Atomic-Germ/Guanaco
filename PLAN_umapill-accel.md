# PLAN_umapill-accel: GPU-pill KV writeback on UMA

## Goal

Add a GPU-pill / unified-memory (UMA) lever so that on integrated/UMA GPUs
(AMD integrated Metal/Vulkan/HIP, Intel integrated), the GPU can compute
prefill and write the KV cache into memory shared with the CPU, which then
reads it back. This is a **prefill-latency lever**, orthogonal to the existing
disk-streaming weight tiering.

Scope is deliberately narrow: this feature exists to make long-context
prefill cheaper where it can be made cheap, and to do so only where the
hardware makes it cheap.

## Non-goal

This is not a discrete-GPU memory optimization. On a discrete GPU the KV
writeback is a PCIe copy that often dominates; there the weight tiering (cold
warm hot) still applies, but the GPU-pill KV writeback does not win. The whole
feature is gated to UMA/integrated devices.

## Governing principles (decided)

1. **Cap-wins.** A hard device-memory cap wins over routing density whenever
   they conflict. Routing density over time decides *within* the cap. The
   warm/cold boundary is itself capped by the RAM cap.
2. **GPU-pill is a latency lever, not a memory lever.** Its job is to shorten
   the prefill tail before TTFT matters. It is orthogonal to tiering and does
   not reduce resident memory.
3. **Discovered threshold.** The GPU-pill wins only past a context length
   where prefill O(n) dominates the GPU transfer + writeback cost. That
   boundary is discovered from routing/pilot density, not hardcoded.
4. **CPU-context-speedup awareness.** On CPU the model is faster at higher
   context (narrower GPU-pill payoff). The prefill tail still wins at the tail;
   keep that distinction in the calibration and the tests.
5. **Compaction as calibration.** A context-compaction routine recalibrates
   the discovered boundaries. It is a calibration lever, not a correctness
   lever.

## Architecture (what to build)

- **Detection (trivial).** Query device properties once. UMA/integrated =>
  feature enabled. Vulkan: `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`. Metal:
  unified memory is implicit. HIP/AMD integrated: report integrated.
- **Gating (trivial).** Auto-enable on UMA only. `--gpu-pill` /
  `LLAMA_GPU_PILL=1` to force; `LLAMA_GPU_PILL=0` to disable. Never enable on
  discrete GPUs.
- **Unified allocation (maybe already present).** Check whether the backend can
  host a host+device-visible buffer for integrated GPUs. If present, this tier
  is zero work; if not, it is moderate and must be verified on-box.
- **Sync ordering (the hard part).** The GPU must signal completion to the CPU
  readback. This is where correctness hides — only real under load. Wire the
  existing ggml/Vulkan sync primitive into the KV flow; do not hand-roll it.
- **Prefill split (moderate).** Split prefill so the GPU owns its chunk and the
  CPU consumes the shared buffer without recomputing. Coordinate so neither
  side double-computes.

## Sensitivity / what to test

- Sync ordering under a live CPU read during a GPU write.
- The warm/cold boundary when both the cap and routing density disagree.
- The discovered prefill threshold at 15%, 60%, 95% of a large context.
- Compaction recalibration does not drift the discovered boundaries.

## Testing discipline

This needs long-running, large-context runs to exercise the sensitive parts.
Anchor every run so it is recoverable:

- branch: `feat/umapill-accel`
- log path: `/tmp/umapill.log`
- run command: a single rerunnable `llama-server --load-mode streaming`
  invocation with fixed context and a fixed prompt corpus
- checkpoint: note the build + patch set that produced each run

Do not start a second run while a sensitive run is in flight; use slots to
isolate them.

## Status

Branch `feat/umapill-accel` exists, empty. No commits yet. This plan is the
start; implementation follows the skill that directs it.
