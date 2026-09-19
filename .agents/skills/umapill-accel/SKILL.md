---
name: umapill-accel
description: Use to implement the GPU-pill / unified-memory (UMA) KV writeback feature on feat/umapill-accel. Covers detection, gating, the sync-ordering hard part, the prefill split, UMA-only scope, and anchored long-context testing.
---

# GPU-pill / UMA KV writeback (branch `feat/umapill-accel`)

## When to use this skill

Use only while working on the GPU-pill / unified-memory feature on the
`feat/umapill-accel` branch. Do not use it for the disk-streaming weight
tiering (cold/warm/hot) — that is separate and orthogonal.

## Anchors (do not change without telling the user first)

- Branch: `feat/umapill-accel` (owned by this skill).
- Upstream checkout: `llama.cpp/` — disposable, never committed, delta shown by
  `git -C llama.cpp diff`.
- Log path for runs: `/tmp/umapill.log`.
- Kill runs by log, not blindly: `pkill -f 'umapill'` or target the PID from
  `/tmp/umapill.pid`.
- Always resume from the anchored log + build, never by re-deriving.

## Read first

Read `PLAN_umapill-accel.md` for the principles and scope. Read it before every
decision; the principles resolve conflicts.

## Rules that govern every change

1. **UMA only.** The feature must never activate on a discrete GPU. Detect the
   device once, gate on UMA/integrated, and skip everything else.
2. **Latency lever, not memory lever.** The GPU-pill shortens the prefill tail
   before TTFT matters. It does not change resident memory. Do not let it
   become a memory optimization.
3. **Cap-wins win.** A hard device-memory cap outranks routing density. The
   warm/cold boundary is itself capped by the RAM cap.
4. **Threshold is discovered.** The context length where the GPU-pill wins is
   derived from routing/pilot density (prefill O(n) vs transfer+writeback), not
   hardcoded.
5. **Reuse ggml sync primitives.** Never hand-roll GPU->CPU sync. The sync
   ordering in the KV flow is the only real correctness risk.

## Implementation order

1. **Detection + gating (trivial, do first).**
   - Detect integrated/UMA from device properties.
   - Auto-enable on UMA only; add `--gpu-pill` / `LLAMA_GPU_PILL=1|0` to force.
   - Verify it stays off on any discrete device.
2. **Unified allocation.**
   - Check if the backend can host a host+device-visible buffer on integrated
     GPUs. If present, reuse it; if not, add minimal support and verify on-box.
3. **Sync ordering (hard part).**
   - Wire the existing ggml/Vulkan sync primitive into the KV flow so the CPU
     readback sees completed GPU writes.
   - Test only under live load; this only bites under real streaming.
4. **Prefill split (moderate).**
   - Split prefill so the GPU owns its chunk and the CPU consumes the shared
     buffer. Ensure neither side double-computes.
5. **Discovery + compaction calibration.**
   - Derive the winning context threshold from routing density.
   - Wire compaction to recalibrate the discovered boundaries, not to fix
     correctness.

## Testing discipline

- Long-running, large-context runs only. Anchor each run: branch,
  `/tmp/umapill.log`, the single rerunnable `llama-server --load-mode streaming`
  command, and the build + patch set.
- Exercise at ~15%, ~60%, ~95% of a large context to see the discovered
  threshold and the CPU-context-speedup narrowing.
- Run at most one sensitive run at a time; use slots to isolate.
- After any change, rerun from the anchored command and diff the log before
  concluding.

## Report

After each run: what changed, the discovered threshold, whether the cap won over
density at the warm/cold boundary, and whether sync held under load. State why
a check could not run instead of assuming it passed.
