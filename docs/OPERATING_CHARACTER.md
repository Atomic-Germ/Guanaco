# Guanaco's operating characteristic

What actually limits throughput, measured rather than assumed. Most of this file
exists because the obvious answers turned out to be wrong.

## The model

For a dense model streamed from disk:

```
t/s  ~=  min( compute_bound ,
               bandwidth / max(0, model_bytes - resident_bytes) )
```

`resident_bytes` is what actually stays in memory as reclaimable page cache.
This is *not* the budget knob - see below.

## Measured

Qwen3.8-27B dense, Q6_K, 19057 MB of weights, 64 layers, single-channel RAM,
NVMe reaching 2.6-2.7 GB/s under concurrent scattered reads. Runs are 16 tokens
with the model's page cache dropped beforehand, inside a cgroup memory limit so
the "available RAM" column is a real constraint rather than a description.

| cgroup limit | resident subset | GB read/token | tok/s | disk rate |
|---------------|-----------------|---------------|-------|-----------|
| 12 GiB | 28/64 | 15.4 | 0.30 | 2.7 GB/s |
| 16 GiB | 38/64 | 10.5 | 0.30 | 2.5 GB/s |
| 20 GiB | 48/64 |  2.1 | 1.20 | 1.1 GB/s |
| 24 GiB | 57/64 |  1.3 | 1.50 | 0.76 GB/s |

The model predicts each row to within ~20%. The knee is sharp: 1.50 tok/s at
24 GiB is exactly the compute-bound figure measured with no memory pressure at
all, and 21 GB total read is one pass over the model.

### The practical rule

**You want roughly 1.4x the model size in RAM for full speed.** A 19 GB model
wants about 27 GB. Below that you are bandwidth-bound and the rate falls as
`bandwidth / (model - resident)`:

| RAM / model | experience |
|-------------|------------|
| >= 1.4x    | compute-bound, ~1.5 tok/s |
| ~1.0x      | knee, transitions fast |
| < 1.0x     | bandwidth-bound, and no tuning changes it |

At 12 GiB with a 19 GB model the throughput is 0.30 tok/s. That is not a
misconfiguration: roughly 10-15 GB must come off the disk every token no matter
how it is configured. If you are there, shorten the context or accept the rate -
do not spend time on prefetch depth.

## The budget knob is a floor, not a throughput lever

`--guanaco-dense-budget` sets a residency *floor* the pager guarantees. It does
not cap what the kernel caches. Evidence:

- peakRSS tracks the cgroup limit, not the budget: 19.4 GB resident under a
  24 GiB limit where the budget was only 17203 MB;
- a run with the budget set to roughly zero (8 MiB, from a units mistake)
  still ran at 1.20 tok/s and read the model exactly once, because the kernel
  could cache all of it.

Raising the fraction above 70% does not help and slightly hurts:

| budget | resident subset | tok/s | GB read |
|--------|-----------------|-------|---------|
| 4000 MiB   | 13/64 | 0.20 | 309 |
| 8601 MiB (auto) | 28/64 | 0.30 | 243 |
| 11500 MiB  | 38/64 | 0.20 | 273 |
| 12500 MiB  | 41/64 | 0.20 | 272 |

A budget too close to the limit converts reclaimable file pages into pages the
kernel has promised to keep, removing its freedom to rebalance against the rest
of the system. 70% is the default for that reason as well as for headroom.

## What the other knobs do

Measured flat in every configuration tried, and they are left at defaults:

- `dense_ahead` (1,2,3,4,8,16,32) - no effect once resident bytes are fixed
- thread count (2,4,6,10) - no effect; the workload is I/O-bound, so CPU has
  headroom by construction
- `pilot_mass`, `max_experts` - MoE; flat, but see the caveat below

## Why the resident subset is only a floor

The pager prints `resident subset: N/64 layers guaranteed ... a floor - the
kernel caches more when RAM allows`. That is deliberate wording. The guarantee is
that N layers will be resident; the kernel will hold more if RAM allows, and may
evict some of them under pressure from the rest of the system. So the curve above
predicts an *upper bound* on speed, not a guaranteed rate.

In the thrashing regime the gap is visible: at 12 GiB, `model - budget` predicts
10.5 GB/token and 15.4 GB/token was measured. The extra ~5 GB is kernel eviction
policy, not a knob.

## Caveats

- One model, one machine. The bandwidth constant is specific to this box, where
  RAM, NVMe and iGPU share a single channel. The shape should generalise; the
  constant will not.
- Dense only. The MoE path routes rather than sweeps, so per-token re-read
  volume is governed by pin-cache hit rate instead, and none of the numbers here
  transfer.
- `resident_bytes` is not directly observable. It is inferred from the cgroup
  limit and the observed read volume.

## Methodology note

Every number above comes from a run that dropped the model's page cache first
and reports `read_bytes` for the process, flagging `[COLD]` or `[WARM]`. That
matters: an earlier sweep of these same knobs produced completely flat results
because the model was already in page cache and the disk was doing nothing.
Anything measured warm is not measuring streaming.

Runs of 8-20 tokens are not enough on their own - the warm-up phase reads far
less than steady state, and short runs will happily show an 18x "improvement"
that is only the difference between loading and thrashing.
