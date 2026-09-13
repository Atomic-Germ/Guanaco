---
name: guanaco-development
description: Use when refreshing llama.cpp upstream, maintaining Guanaco patches, reading PLAN_* project files, starting Guanaco work, building, testing, branching, or preparing a Guanaco patch set.
---

# Guanaco Development

Follow the root `AGENTS.md` as the authoritative end-to-end workflow.

Start with its upstream-refresh phase. After the refreshed patch set is built
and ready on `main`, read every root `PLAN_*` file, present the available
projects, and ask the user which one to undertake. Do not select a project or
create its typed topic branch without that answer.

Keep `llama.cpp/` disposable and uncommitted, represent its full delta in
`patches/*.patch`, verify clean patch application and the root build, and obtain
explicit approval immediately before any commit or push.
