# Design notes

Why the pieces look the way they do, including what was tried and dropped.

## Event schema

Every collective carries `(pg_id, seq_id)`: the process group and its per-group sequence number, the same identifiers PyTorch's NCCL flight recorder records. Every event carries `rank`, `node_id`, `gpu_uuid` and a per-rank host timestamp.

## Why (pg, seq) grouping, not timestamps

The first scaffold planned to group collectives across ranks by timestamp proximity (~1 ms). That needs synchronised clocks across nodes (PTP-grade), and it fails silently when they drift. Grouping by `(pg, seq)` is exact and needs no clocks. Timestamps are then used only within one rank, to join that rank's own hardware samples to its own collectives.

## Why lag instead of arrival time

In a synchronising collective, ranks that arrive early wait; the last to arrive waits least. So for one collective, `lag_r = max(duration) - duration_r` is largest for the rank that arrived last. It is computed from durations alone, so per-rank clock offsets cancel. `TestPerRankClockSkewDoesNotBreakDetection` pins this.

## Why block medians, not per-collective thresholds

First implementation: flag a collective if a rank's lag has a robust z-score above a threshold, then require repeats. Result on the simulator: perfect at 3% duration jitter, **0/30** at 10% jitter, because a 3 ms lag is only ~3 sigma of a single noisy collective. But a persistent 3 ms shift over hundreds of collectives is statistically obvious. So evidence now accumulates: per block of 30 collectives, compare each rank's *median* lag with its peers', scaled by the standard error of a median. Noise is estimated with MAD from the *other* ranks, so the straggler cannot inflate its own baseline (this was a real bug at 2 ranks). Result: 30/30 at 10% jitter, and a clear floor near 5% lag.

Persistence is judged over the extent between a rank's first and last flagged block, not to the end of the window, so a fault that has since recovered is still reported (found by `TestWindowLimitsAnalysis`). For a flight recorder, the post-mortem is the use case.

Known limits: blocks are non-overlapping, so onset is located to within ~one block; a trailing partial block is ignored; needs at least two blocks (60 collectives).

## Attribution

Rules over the straggler's own telemetry, each returning (fired, had-data). Three properties matter more than coverage:

1. A rule with no data neither fires nor claims "ruled out".
2. Bandwidth and clock rules need a pre-window baseline; without one they stay silent.
3. Confidence is capped below 1 and explicit throttle rules outrank inferred ones (`clock_reduced`).

Host stall is suppressed when a hardware throttle is present: a throttled GPU that looks idle is a device problem, not a starved dataloader.

GPU contention is detected through the NVML compute-process count rising above its pre-fault baseline. A count of 0 means *unknown* (unsupported, or a container hides other PIDs), never "no processes", so the rule stays silent and the result is `unknown`. The simulator scores both the visible and the hidden case.

Not covered: network fabric faults, cross-node effects, and contention when the process list is hidden.

## Collectors

The first scaffold planned to parse `NCCL_DEBUG=TRACE` text logs. That output does not carry per-collective durations, so the plan was dropped in favour of PyTorch's flight recorder (which does, with `TORCH_NCCL_ENABLE_TIMING=1`). NVML is sampled in the Python shim, in the rank's own process, so the Go agent only tails JSONL and needs no cgo.

## Ring buffer

Mutex-guarded, power-of-two capacity, overwrite-oldest. A lock-free single-producer/single-consumer design cannot overwrite safely because the producer would have to advance the consumer's read cursor. Push costs ~6 ns with no allocation.
