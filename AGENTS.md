# Guanaco Development Instructions

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

- `llama.cpp/`: a clone of `https://github.com/ggml-org/llama.cpp`.
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
