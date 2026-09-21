# GPU Flight Recorder

[![ci](https://github.com/RyanJHamby/distributed-gpu-training-flight-recorder/actions/workflows/ci.yml/badge.svg)](https://github.com/RyanJHamby/distributed-gpu-training-flight-recorder/actions/workflows/ci.yml)

**Finds the straggler in a distributed training job and explains why it is slow.**

In data-parallel training, one slow rank makes every other rank wait at every collective. The symptom (step time went up) says nothing about the cause: a thermal or power throttle, an ECC fault, a degraded PCIe or NVLink link, or a starved dataloader. Flight Recorder joins per-rank collective timing with per-GPU hardware telemetry, names the straggling rank, and ranks the likely causes with the evidence behind each.

```
$ gfr replay examples/traces/thermal.jsonl
3840 events, 8 ranks, 1 anomalies
  rank 2 straggled on AllReduce (150/150 collectives, ~3.0ms lag): thermal_throttle
  (confidence 0.75) - thermal throttle active in 30/40 samples; also present: clock_reduced
```

## Status: what is and is not validated

This section comes first on purpose.

| | State |
|---|---|
| Detection, attribution, transport, agent, CLI | Implemented. Unit, fuzz and end-to-end tests run under `-race` in CI |
| Simulator with injected ground-truth faults | Implemented; scorecard below |
| Python shim (PyTorch NCCL flight recorder + NVML to JSONL) | Implemented and tested against fakes. **Never run against real PyTorch or NVML**; the entry field names it reads are unverified |
| Real-GPU harness ([`bench/`](bench/README.md)) | Drafts. Only the scorer is tested (on simulator traces) |
| Real-hardware results | **None yet.** Every accuracy number in this repo is synthetic |
| Docker image and compose demo ([`deploy/`](deploy/)) | Built and run: 8 containerized agents stream a replayed trace to a coordinator and `gfr report` names the straggler |
| Kubernetes manifests | Parse-checked only; never applied to a cluster |

The next milestone is a budget-capped real-GPU run (runbook in `bench/README.md`). A result that disagrees with the synthetic scorecard will be published, not tuned away.

## Quickstart (no GPU needed)

```bash
make build                                    # needs Go 1.25+
./bin/gfr replay examples/traces/thermal.jsonl
./bin/gfr replay examples/traces/clean.jsonl  # 0 anomalies

# The distributed shape: one coordinator, one agent process per rank, over gRPC
./bin/gfr coordinator --listen :50051 &
pids=()
for r in 0 1 2 3 4 5 6 7; do
  ./bin/gfr agent --coordinator localhost:50051 --replay examples/traces/thermal.jsonl --ranks $r &
  pids+=($!)
done
wait "${pids[@]}"                             # agents exit when their replay ends
./bin/gfr report --coordinator localhost:50051
kill %1                                       # stop the coordinator

make sim      # regenerate the scorecard (docs/scorecard.md)
make traces   # write one synthetic trace per fault type into ./traces
docker compose -f deploy/docker-compose.yaml up -d   # same demo, containerized; then: gfr report
```

## How it works

```
 training process (per rank)                     node                 cluster
┌────────────────────────────────────┐   ┌──────────────────┐   ┌──────────────────────────────┐
│ PyTorch NCCL flight recorder ──┐   │   │      agent       │   │        coordinator           │
│                                ├─▶ JSONL ─▶ tail collector │   │                              │
│ NVML sampler (clocks, throttle,│   │   │      │           │   │  correlator ──▶ attribution  │
│ ECC, PCIe, temp, power) ───────┘   │   │  ring buffer     │   │  (straggler)     (why)       │
│  shim/gfr_torch.py                 │   │      │ gRPC      │──▶│        │                     │
└────────────────────────────────────┘   └──────────────────┘   │     report (JSON / CLI)      │
                                                                └──────────────────────────────┘
```

Four decisions carry most of the design. Each has a written rationale and, where relevant, a measured before/after in [docs/design.md](docs/design.md).

**1. Correlate by `(process_group, sequence_id)`, not by timestamp.** Collectives are matched across ranks by the identifiers PyTorch already records. No cross-node clock synchronisation is needed, and a drifting clock cannot silently mis-group events.

**2. A skew-free lag signal.** In a synchronising collective the rank that arrives last waits least, so `lag = max(duration in group) - own duration` is largest for it, and it is computable from durations alone.

```
 rank 0  |=====wait=====[  allreduce  ]|   duration 13 ms   lag  0
 rank 1  |===wait===[  allreduce  ]    |   duration 11 ms   lag  2
 rank 2  |[  allreduce  ]              |   duration 10 ms   lag  3   <- arrived last: straggler
         t0       arrival order       t1
```

**3. Evidence accumulates; single collectives are not thresholded.** The first version thresholded each collective and scored **0/30** at 10% duration jitter. Detection now cuts collectives into blocks of 30 and compares each rank's *median* lag against its peers', scaled by the standard error of a median. Noise is estimated from the *other* ranks, so a straggler cannot inflate its own baseline. Same simulator, 30/30.

**4. Attribution is explainable rules, not a model.** Each rule inspects the straggler's own telemetry around the fault onset and returns *fired* and *had data*. Three properties matter more than coverage:
- A rule with no data neither fires nor claims the cause is ruled out.
- Bandwidth and clock rules need a pre-fault baseline; without one they stay silent.
- Confidence is capped below 1, and explicit throttle evidence outranks inferred clock drops.

| Cause | Evidence |
|---|---|
| `thermal_throttle`, `power_throttle` | NVML throttle-reason bitmask |
| `clock_reduced` | SM clock fell >25% vs baseline (catches locked clocks and unreported throttles) |
| `gpu_contention` | NVML compute-process count on the GPU rose above its pre-fault baseline (silent if PIDs are hidden) |
| `ecc_errors` | corrected / uncorrected counter increase over the window |
| `pcie_bandwidth_drop`, `nvlink_degraded` | throughput < 50% of pre-fault baseline |
| `memory_pressure` | sustained memory-bandwidth saturation |
| `host_stall` | GPU mostly idle while the rank lags; suppressed if a hardware throttle explains it |
| `unknown` | nothing fired; the chain lists what was checked and ruled out |

Example report entry (`gfr replay --json`):

```json
{
  "anomaly": { "straggler_rank": 2, "op_type": "AllReduce", "lag_ns": 3024080,
               "deviation_sigma": 25.5, "hits": 150, "groups": 150 },
  "causal_chain": [
    { "cause": "thermal_throttle", "confidence": 0.75, "detail": "thermal throttle active in 30/40 samples" },
    { "cause": "clock_reduced",    "confidence": 0.61, "detail": "SM clock fell 42% vs baseline (1900 -> 1100 MHz)" } ],
  "ruled_out": ["power_throttle", "ecc_errors", "pcie_bandwidth_drop", "nvlink_degraded", "memory_pressure", "host_stall"]
}
```

## Results (synthetic)

`make sim` reproduces [docs/scorecard.md](docs/scorecard.md): 8 ranks, 600 steps, one fault injected on a random rank at the midpoint, 30 seeds per cell, seeded and deterministic.

| Condition | Detected | Attributed correctly | False stragglers |
|---|---|---|---|
| 9 fault types, 3% op jitter, 30% straggler lag | 30/30 each | 30/30 each | 0 |
| 9 fault types, 10% op jitter, 30% straggler lag | 30/30 each | 30/30 each | 0 |
| Clean runs (both noise levels) | n/a | n/a | 0 |

Sensitivity of thermal-fault detection to straggler lag (3% jitter): **2% lag: 0/30. 5% lag and above: 30/30.** Below about 5% of op duration a shift is treated as noise, by design.

Read these numbers with the caveats: the simulator's noise is i.i.d. Gaussian, real clusters are correlated and heavy-tailed, and each attribution rule is tested against a fault the simulator generated to trigger it. This shows the logic is sound, not that it works on hardware. Two contention rows are included on purpose: with visible process counts the tool reports `gpu_contention`; with PIDs hidden (typical inside containers) it locates the straggler and reports `unknown` rather than guess.

## Scope and how this relates to other tools

**In scope:** persistent stragglers in synchronous data-parallel training on a NVIDIA GPU cluster, located to a rank and attributed to a per-GPU cause, with post-mortem reporting after the fault ends.

**Out of scope:** hangs and deadlocks; network-fabric or switch faults; cross-node causes; non-NVIDIA hardware; anything needing per-kernel detail.

| Tool | What it gives you | What this adds |
|---|---|---|
| PyTorch NCCL flight recorder | Per-rank collective history, mostly consumed for hangs and mismatches | It is this project's *input*. Adds cross-rank straggler detection and hardware attribution |
| DCGM / NVML dashboards | Per-GPU health and telemetry | No notion of which rank held up which collective. This joins the two |
| Nsight Systems / profilers | Deep single-run kernel-level detail | Heavy and run-scoped. This is meant to stay on, and answers "which rank and why" first |

## Repository layout

```
cmd/
  gfr/               CLI: agent | coordinator | replay | report
  gfr-sim/           scorecard and synthetic-trace generator
internal/
  types/             event schema and config
  buffer/            overwrite-oldest ring buffer (mutex, power-of-two)
  correlator/        (pg, seq) grouping, MAD stats, block-median straggler detection
  attribution/       rule-based causal chain with confidence and ruled-out list
  coordinator/       correlator -> attribution pipeline; live (streaming) and offline
  transport/         gRPC server/client, proto <-> domain conversion
  agent/             collector fan-in, ring buffer, forwarding
  collector/         replay (JSONL trace) and tail (follow shim output) collectors
  trace/             JSONL trace format
  sim/               synthetic fault generator and scoring
api/proto/           wire schema and generated code (committed)
shim/                Python: PyTorch flight recorder + NVML -> JSONL, with tests
bench/               real-GPU harness: workload, fault injector, scorer, matrix runner
examples/traces/     two small recorded traces used in the quickstart
testdata/shim/       fixture produced by the Python shim; pins the Python/Go contract
deploy/              Dockerfile, compose demo (verified), Kubernetes manifests (unrun)
docs/                design.md (rationale), scorecard.md (generated)
```

## Testing strategy

- **Unit and property tests** for every package, run with `-race`; ring-buffer concurrency test and benchmark (~6 ns/push, no allocations).
- **Fuzzing** of correlator ingest and detection against out-of-order, duplicate and partial events.
- **Bug-driven regression tests.** Several came from failures the simulator exposed: a straggler contaminating its own baseline at 2 ranks; persistence diluted by a healthy prefix; a recovered fault being dropped.
- **Cross-language contract test:** a fixture written by the Python shim must decode in Go, so field-name or unit drift fails CI.
- **End to end:** 8 agents stream to one coordinator over gRPC and the report is checked; a multi-process run of the real binaries is in the quickstart.
- **Scorecard as a regression floor:** `internal/sim` fails if detection or attribution drops.

## Limitations

- No real-hardware validation yet (see Status).
- Detection needs at least two blocks (60 collectives) of history and a persistent fault; onset is located to about one block. Single slow steps are ignored on purpose.
- Attribution is per-GPU. It cannot see fabric faults, and it can only explain GPU sharing when NVML exposes the process list.
- The gRPC channel is plaintext (development only); no auth or TLS yet.
- The coordinator holds events in memory (bounded, oldest dropped) and is single-instance.

## Roadmap

1. Real-GPU smoke session: confirm the flight-recorder schema, fix the shim, publish a first hardware scorecard.
2. Confirm on real hardware that NVML process counts are visible in the target container setups.
3. TLS/mTLS on the agent-coordinator channel; overlapping blocks for finer onset.
4. Overhead measurement (step time with and without the recorder).

## Development

```bash
make test     # go test -race ./...
make proto    # regenerate api/proto (protoc, protoc-gen-go, protoc-gen-go-grpc)
make sim      # regenerate docs/scorecard.md
python -m pytest shim/tests bench/tests
```

MIT licensed.
