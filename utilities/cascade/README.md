# Cascade: quick and considered

Two models, one always resident and fast, one slower and better on the work that
matters. The second is loaded on demand or kept resident, whichever suits how
often you actually use it.

## Why the roles are not "small" and "large"

Parameter count does no work in this design. The 24B-A8B MoE has 8B active, runs
3.3x faster than the 27B dense, and is *less* capable one-shot - so it is the
**quick** model despite being the larger file on disk. Capability and
cost-per-token are independent axes, and a model is assigned a role explicitly.
`--considered considered_moe` swaps which model plays considered without
touching anything else.

## What this is not

Not speculative decoding. That trades samples for **speed**: a cheap model
drafts, an expensive model verifies, same answer set, faster. This trades
samples for **capability**: a cheap model answers, and sometimes that answer is
replaced by one from a better model. The cascade makes one escalation decision
per request and gets no second sample.

That difference is why the models are chosen the way they are. A cascade needs
the considered model to be more correct *one-shot*, because there is no
best-of-n and no verification pass. Observed on this machine: the 27B dense is
markedly more reliable in single attempts than the 8B, which tends to answer
circularly and then repeat itself.

The clearest example, same question, both one-shot:

> **8B:** "Raft is safe because it uses a distributed consensus algorithm (Raft)
> to ensure that all nodes agree on the state of the system..." then repeats it.
>
> **27B dense:** "Raft is safe because it enforces a single leader who must
> obtain a majority of votes before committing any log entry, ensuring that all
> replicas agree on the same sequence of operations."

## Measured

27B dense / 8B, Qwen3.8-27B-antislop Q6_K, AesCode-8B Q4_K_M. Absolute rates on
this box drift with thermal state, so only ratios are trustworthy; the raw
figures below are indicative.

| | tok/s |
|---|---|
| quick (8B) alone | ~6.4 |
| quick while the considered model pages | ~6.4 (no measurable cost) |
| considered, dense 27B | ~1.3 |
| considered, MoE 24B-A8B | ~4.4 |

Blended rate if a fraction *f* escalates: `1 / (f/r_quick + (1-f)/r_considered)`.

| pairing | f=0.5 | f=0.8 | gain over considered alone |
|---|---|---|---|
| 8B quick + 27B dense | 4.2 | 3.6 | **1.9x** |
| 8B quick + MoE | 5.2 | 4.6 | 1.18x |

The cascade is worth far less when the considered model is a fast MoE. **Its
value is inversely proportional to how good the considered model is** - which is
why picking that model matters more than any escalation policy.

## Resident vs on-demand

The considered model costs one of two things: a permanent ~25% tax on the quick
model's throughput, or a load per use.

| | quick | considered |
|---|---|---|
| resident | ~6.4 tok/s always | instant |
| on-demand | ~6.4 tok/s always | ~20 s warm, ~40-70 s cold, per use |

On-demand wins unless the considered model is used more than about once per
40-130 quick tokens. That break-even is a property of the workload, not
something this tool decides, which is why it is a switch rather than a policy.

    python3 cascade.py --mode on-demand  ask considered "..."
    python3 cascade.py --mode resident   up

## CPU-bound, deliberately

Both roles run with `-ngl 0`. llama-server with any Vulkan-backed KV cache
segfaults on the second request:

    server_slot::prompt_save
    llama_context::state_seq_get_data
    llama_io_write_host
    ggml_backend_vk_buffer_get_tensor
    ggml_vk_buffer_read_2d              <- SIGSEGV

Reproduced 4/4 with identical stacks, with gpu-pill **on and off**, so it is not
Guanaco's UMA allocation - upstream `read_2d` faults whenever the KV cache is in
a Vulkan buffer. `attrib.sh` reproduces the attribution. Costs about 6% on the
quick model versus gpu-pill's 7.6 tok/s.

## Usage

    python3 cascade.py up                       # quick model only
    python3 cascade.py ask quick   "..."
    python3 cascade.py ask considered "..."     # loads it, answers, unloads
    python3 cascade.py serve                    # q: quick, c: considered
    python3 cascade.py status
    python3 cascade.py down

Model paths come from `model_8b.txt`, `model_path.txt`, `model_moe.txt` beside
this script - one GGUF path per line. Edit those to point at your own models.

## Methodology

`bench.py` drops the model's page cache before every run and reports
`read_bytes` from `/proc/<pid>/io`, flagging `[COLD]` or `[WARM]`. This matters:
an earlier sweep of these same knobs produced entirely flat results because the
model was already cached and the disk was doing nothing.

Absolute tok/s on this machine drifts by 25% or more with thermal state - the
same 8B measured 7.4 tok/s in one session and 5.7 in another with nothing else
running. Compare conditions **interleaved**, within one session. Runs of 8-20
tokens are not enough on their own: the warm-up phase reads far less than steady
state, and a short run will show an 18x "improvement" that is only the
difference between loading and thrashing.
