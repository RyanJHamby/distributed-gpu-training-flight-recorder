# GPU Flight Recorder

Finds the straggler in a distributed training job and says *why* it is slow.

When one rank in a data-parallel job is slow, every other rank waits at each collective. The symptom (step time goes up) says nothing about the cause: a thermal or power throttle, an ECC fault, a degraded PCIe/NVLink link, or just a starved dataloader. Flight Recorder correlates per-rank collective timing with per-GPU hardware telemetry, names the straggling rank, and ranks the likely causes with the evidence behind each.

## Status

Read this first. It is honest about what runs.

| Component | State |
|---|---|
| Ring buffer, correlator (straggler detection), attribution engine | Implemented, tested (`go test -race`), fuzzed |
| JSONL trace format, `gfr replay` | Implemented |
| gRPC agent → coordinator streaming, `gfr report` | Implemented; tested with 8 agent processes on one coordinator |
| Synthetic fault simulator and scorecard | Implemented, results below |
| Live NVML collector, PyTorch NCCL flight-recorder shim | **Not yet implemented** |
| Validation on real GPUs | **Not yet done.** All numbers below are synthetic |

Today the tool works from recorded or synthetic traces. The agent's only collector is `replay`; nothing reads a real GPU yet.

## How it works

```
 rank 0 agent ─┐
 rank 1 agent ─┼─ gRPC stream ─▶ coordinator ─▶ correlator ─▶ attribution ─▶ report
 rank N agent ─┘   (events)        (per-(pg,seq)   (block-median   (rule-based,
                                     grouping)       straggler       ranked causes
                                                     test)           + evidence)
```

**Correlation key.** Collectives are grouped across ranks by `(process_group, sequence_id)`, the same identifiers PyTorch's NCCL flight recorder emits. No timestamp matching, so no cross-node clock synchronisation.

**Skew-free signal.** In a synchronising collective the last rank to arrive sees the *shortest* duration, so `lag = max(duration in group) − own duration` needs no clocks at all.

**Detection.** Per-collective thresholds are brittle under noise, so evidence accumulates instead. Collectives are cut into blocks of 30. In each block a rank's median lag is compared with the other ranks', scaled by the standard error of a median (noise estimated with MAD, excluding the rank under test so a straggler cannot inflate its own baseline). A rank is reported when flagged persistently.

**Attribution.** Explainable rules over the straggler's own telemetry around the onset: thermal throttle and power cap (NVML throttle-reason bitmask), ECC counter deltas, PCIe and NVLink throughput drop against a pre-fault baseline, memory-bandwidth saturation, and host stall (GPU idle while the rank lags, suppressed when a throttle explains it). Output is a ranked chain with confidence, the rules that were checked and ruled out, and a summary. Nothing fires without data: a bandwidth rule with no baseline neither fires nor claims to have ruled the cause out.

## Try it (no GPU needed)

```bash
make build
go run ./cmd/gfr-sim -write-traces testdata      # or use the committed traces
./bin/gfr replay testdata/thermal.jsonl
# 3840 events, 8 ranks, 1 anomalies
#   rank 2 straggled on AllReduce (150/150 collectives, ~3.0ms lag): thermal_throttle (confidence 0.75) ...
./bin/gfr replay testdata/clean.jsonl            # 0 anomalies

# Distributed shape: one coordinator, one agent process per rank
./bin/gfr coordinator --listen :50051 &
for r in 0 1 2 3 4 5 6 7; do ./bin/gfr agent --coordinator localhost:50051 --replay testdata/thermal.jsonl --ranks $r & done
wait; ./bin/gfr report --coordinator localhost:50051
```

## Results (synthetic)

`make sim` regenerates [docs/scorecard.md](docs/scorecard.md): 8 ranks, 600 steps, a fault injected on a random rank at the midpoint, 30 seeds per cell.

- 3% and 10% op-duration jitter with a 30% straggler lag: every fault type detected and attributed correctly in 30/30 runs, with 0 false stragglers on clean runs.
- Sensitivity floor: a thermal fault is detected 30/30 at 5% lag and 0/30 at 2% lag. Below about 5% of op duration the shift is treated as noise, by design.
- A straggler whose cause leaves no per-GPU hardware signal (GPU shared with another process) is reported as `unknown`; the tool locates it but does not explain it. Explaining it needs per-process NVML data.

## Limitations

- Synthetic noise is i.i.d. Gaussian; real clusters have correlated, heavy-tailed noise. Expect worse numbers on hardware until validated there.
- Attribution is rule-based and per-GPU. It does not see network fabric faults or cross-node effects.
- Detection needs at least two blocks (60 collectives) of history and a persistent fault; single slow steps are ignored on purpose.
- No authentication or TLS on the gRPC channel yet (development only).
- NVIDIA only.

## Development

```bash
make test          # go test -race ./...
make proto         # regenerate api/proto/*.pb.go (needs protoc, protoc-gen-go, protoc-gen-go-grpc)
make sim           # regenerate docs/scorecard.md
```

## License

MIT
