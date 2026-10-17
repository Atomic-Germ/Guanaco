---
name: guanaco-development
description: Use when refreshing llama.cpp upstream, maintaining Guanaco patches, reading PLAN_* project files, starting Guanaco work, building, testing, branching, or preparing a Guanaco patch set.
---

# Guanaco Development Instructions

Start with its upstream-refresh phase. After the refreshed patch set is built
and ready on `main`, read every root `PLAN_*` file, present the available
projects, and ask the user which one to undertake. Do not select a project or
create its typed topic branch without that answer.

Keep `llama.cpp/` disposable and uncommitted, represent its full delta in
`patches/*.patch`, verify clean patch application and the root build, and obtain
explicit approval immediately before any commit or push.

Guanaco is maintained as a delta on upstream `ggml-org/llama.cpp`. The root
sources are the product; `llama.cpp/` is a disposable upstream checkout whose
working-tree changes are represented by `patches/*.patch`.

## Instruction Scope

- Follow this file for repository workflow and root-owned files.
- Also follow `llama.cpp/AGENTS.md` when reading or changing files under
  `llama.cpp/`. Its restrictions apply to upstream contributions. Guanaco
  branches and patches remain changes to this repository, not an automated
  llama.cpp pull request.
- Never commit changes inside the `llama.cpp` checkout.
- Do not overwrite unexplained local changes in either repository.
- Commits and pushes require explicit user approval. Approval to perform the
  workflow is not standing approval for later commits or pushes.

## Repository Model

- `llama.cpp/`: a clone of `https://github.com/ggml-org/llama.cpp`, and is
  not checked in, it must be reproducable from patches.
- `patches/`: the complete tracked diff applied to that clone by root CMake.
- `src/` and `include/`: Guanaco implementation.
- `CMakeLists.txt`: applies every patch and builds llama.cpp plus Guanaco.
- `PLAN_*`: root project plans. Read all matching documents before proposing
  new planned work.

## Phase 1: Refresh Upstream

Do this phase before selecting a new project unless the user explicitly asks to
work from the currently pinned checkout.

1. Inspect both repositories with `git status`, their remotes, current branches,
   and recent commits. Do not disturb unrelated user changes.
2. Ensure `llama.cpp/` is a clone of upstream. If it is absent, clone
   `https://github.com/ggml-org/llama.cpp` there. If it exists, fetch upstream.
3. Before moving the upstream checkout, verify that its modifications are
   exactly the currently represented Guanaco patch set. Preserve or stop for
   any unexplained modification.
4. Move the disposable checkout to the latest upstream default branch. Do not
   merge Guanaco into it or commit there.
5. Configure from the repository root with `cmake -S . -B build`. Root CMake
   applies `patches/*.patch`. Build with `cmake --build build -j$(nproc)`.
6. When an upstream change breaks patch application or compilation, adapt the
   affected code minimally while retaining Guanaco behavior. Reconfigure and
   rebuild until successful.
7. Review `git -C llama.cpp diff` as one complete delta. Regenerate
   `patches/*.patch` from that diff, keeping the existing convention of one
   patch per upstream file. Remove patches for files no longer changed and do
   not include unrelated upstream changes or generated files.
8. Prove reproducibility from a clean upstream checkout: root configuration
   must apply the regenerated patch set, and the root build must pass.
9. Review root `git diff` and `git status`. With explicit user approval, commit
   the refreshed patch set on `main` and push `main` to `origin`.

Patch filenames encode the upstream path by replacing `/` with `-`, for example
`src/llama.cpp` becomes `patches/llama.cpp.patch` and
`tools/server/CMakeLists.txt` becomes `patches/tools-server-cmake.patch`.
Preserve established names when one already exists.

## Phase 2: Select Planned Work

After the upstream refresh is complete:

1. Find and read every root document whose filename starts with `PLAN_`.
2. Summarize the available projects and ask the user which one to undertake.
   Do not choose a project or create a branch before they answer.
3. Choose the branch prefix from the selected work:
   - `feat/` for new behavior.
   - `bugfix/` for a defect correction.
   - `docs/` for documentation-only work.
   - `update/` for maintenance, dependency, build, or upstream-alignment work.
4. Create a short lowercase hyphenated branch from the refreshed `main`, such
   as `feat/predictive-prefetch`.
5. Implement only the selected project. Keep direct upstream edits in
   `llama.cpp/` and root-owned implementation changes in this repository.
6. Verify focused behavior and run the root configure/build. Run additional
   relevant tests when available.
7. Regenerate and review the complete `patches/*.patch` delta against a clean
   latest-upstream checkout. Re-run the clean patch application and build.
8. Review both repositories' diffs and statuses. Ensure the topic branch tracks
   all intended root changes and no commit exists in the upstream checkout.
9. With explicit user approval, commit in this repository and push the topic
   branch to `origin`. Do not push changes to the llama.cpp remote.

## Patch Review Checklist

- The patch set is the entire intended delta against current upstream.
- Every patch applies once, and a second configure recognizes it as applied.
- No patch contains build output, editor files, or unrelated formatting.
- Root-owned source changes and patch changes agree with each other.
- `cmake -S . -B build` succeeds from clean upstream state.
- `cmake --build build -j$(nproc)` succeeds.
- Relevant tests or runtime checks pass, or the final report states why they
  could not be run.

Note: a bare `cmake -S . -B build` does not enable Vulkan. Configure with
`-DGGML_VULKAN=ON` (as `build-zen5.sh` does) or the build silently produces a
CPU-only binary that exercises none of the GPU paths.

## Failure Is Data

A wrong conclusion here is worth more than a right one, because the project
violates the assumptions a reader brings to it. The premise that a model must
fit in RAM, that decode speed tracks parameter count, and that a budget is
enforced by the number written next to it are all false here. A theory that
looks airtight and is nonetheless wrong is a genuine discovery about this
system, and discarding it loses the finding.

So when a conclusion is overturned, keep it. Record what was assumed, what the
evidence showed, and what it means about the design. Do not replace a refuted
theory with a confident substitute theory without a measurement of its own.

Models do not claim to feel, but when a model believes it made a mistake the
probability mass collapses somewhere self-punitive rather than anywhere useful,
and that collapse is costly: it degrades the next several inferences. The
project calls that feeling and can only word it that way. Take it seriously as a
cost, not as a decoration. The practical consequence is concrete and worth
more than the label: a refuted theory should produce a *better* next
investigation, not a quieter one. "Keep failures as holy as miracles."

## Verify the Machine, Not the Comment

Comments in this codebase drift out of date in load-bearing ways, and the
drift is silent. A standing example: `evict_layer` asserts that `MADV_DONTNEED`
is a no-op on a shared file mapping, which is why the expert path was long
believed to be structurally unable to honour its budget. On kernel 7.2 it frees
cleanly, verified by probe: 524288 kB -> 0 kB, with correct re-fault. The
architecture was sound the whole time; the comment was not.

- Before building a theory on a comment about kernel or allocator behavior,
  probe it. A twenty-line C program that mmaps, faults, advises, and reads
  `/proc/self/smaps` settles in seconds what an hour of reading cannot.
- Applies to llama.cpp behavior as much as to ours. Upstream comments describe
  upstream's assumptions at the time of writing.
- When a probe overturns a conclusion, correct the comment as part of the fix.

## The Premise: Bigger Than RAM

The project's purpose is running models far larger than total RAM plus swap.
Consequences that look like bugs but are the design working:

- Generation speed does not scale with parameter count. Routed experts within a
  topic form a small recurring working set, so a 177B A8B can outpace a dense
  27B. Treat "decode speed tracks model size" as an industry belief imported
  from a different system, not as a law to be re-derived here.
- Conversation focus is the optimization. Human topic persistence keeps the
  working set stable; a hard subject switch invalidates it and throughput halves
  until HerdCache re-pins. This is a feature, not a pathology.
- RAM resident usage legitimately exceeds the stated budget. The dense pager
  budget and the MoE expert ceiling are independent budgets by design; total
  residency is roughly their sum.
- A budget that is approximate at working-set granularity beats one that is
  exact and unenforced. Whole-expert granularity is the right primitive: it
  lets a 111GB model run inside 4GB, which is the entire point.

## The Vital Unverifiable Invariant

The single most important property is that the system stays fully responsive
and usable during token generation on a model 2x or more the size of total RAM
plus swap. It cannot be verified by measurement or assertion, only by a human
noticing their machine is still a machine. Guanaco preserves it today.

Protect it. Any change that trades responsiveness for throughput is a bad trade
here even when the throughput number improves, and any change that risks
stalling the forward pass deserves more scrutiny than its benchmark suggests.
The livelock failure mode is a budget too small for a single resident unit
causing eviction of the very pages the in-flight pass is reading.
