---
id: gfr-sim
goal: A deterministic, seeded fault simulator exists in internal/sim; the same Config always yields byte-identical per-rank events with one documented injected fault (or none) and ground truth, labelled as simulated data.
acceptance:
  - go test -tags spec_gfr_sim ./internal/sim/... -count=1
  - sh -c 'test -z "$(grep -l spec_gfr_sim internal/sim/sim_test.go)"'
  - go test ./internal/sim/... -count=1
  - go vet ./internal/sim/...
scope:
  - internal/sim/sim.go
  - internal/sim/sim_test.go
non_goals:
  - do not change any assertion in internal/sim/sim_test.go; the only permitted edit is deleting its first line (`//go:build spec_gfr_sim`) and the blank line after it, which is what makes the third command run the test in normal CI
  - no changes to internal/correlator, internal/attribution, internal/buffer or internal/types (no detector logic, no new event types or fields)
  - no new third-party dependencies; use only the standard library (math/rand with rand.New(rand.NewSource(seed)), never the global source, never time.Now)
  - no GPU, NVML, DCGM, nvidia-smi or network calls; do not touch scripts/inject-fault.sh
  - no clock skew between ranks and no sequence ids (types.NCCLCollectiveEvent has none); that is a separate gap
  - do not write results files or quote any measured number anywhere
risk_tier: 2
max_runs: 3
autonomy: inherit
status: draft
---

## Why
The repo has no way to produce a reproducible straggler scenario without a GPU (`scripts/inject-fault.sh` is a stub that
needs root and `nvidia-smi`). A simulator gives the correlator and attribution engine something deterministic to be
measured against later (see spec `gfr-eval`). Everything it produces is simulated data and must say so.

## Where
New package `internal/sim` (file `internal/sim/sim.go`, package name `sim`). Module path:
`github.com/ryanhamby/gpu-flight-recorder`. Event types come from `internal/types` and are appended as POINTERS
(`*types.GPUMetricEvent`, `*types.ThermalEvent`, `*types.NVLinkEvent`, `*types.NCCLCollectiveEvent`), the same way
`internal/buffer/ringbuffer_test.go` does. The test is already written: `internal/sim/sim_test.go` (build tag `spec_gfr_sim`).
Read it first; it is the contract.

## API (exact names the test uses)
```go
type FaultType string   // FaultNone "none", FaultThermalThrottle "thermal_throttle", FaultECCErrors "ecc_errors",
                        // FaultNVLinkDegraded "nvlink_degraded", FaultMemoryPressure "memory_pressure", FaultHostStall "host_stall"
type JitterLevel string // JitterLow "low" (sigma 0.01), JitterMedium "medium" (0.03), JitterHigh "high" (0.08)
const DataLabel = "simulated"
const BaseNs = int64(1_000_000_000); StepIntervalNs = int64(10_000_000)
const BaseDurationNs = 1_000_000.0 // untyped/float constant: healthy collective duration (ns)
const NominalNVLinkGBs = 300.0

type Config struct { Seed int64; Ranks, Steps int; Fault FaultType; FaultRank uint32; FaultStartStep, FaultSteps int; Jitter JitterLevel }
type Truth struct { HasFault bool; Fault FaultType; Rank uint32; FirstStep, LastStep int; ExpectedCause string } // steps inclusive
type Scenario struct { Config Config; Data string; Events []types.Event; Truth Truth } // Data == DataLabel
type FaultSpec struct { Type FaultType; Description, ExpectedCause, Signature string }
func Catalogue() []FaultSpec                  // the 5 faults (not "none"), sorted by Type, all fields non-empty
func Generate(cfg Config) (Scenario, error)
```
`BaseDurationNs` must be usable in `float64(x) > 1.5*sim.BaseDurationNs` (an untyped float constant works; `NominalNVLinkGBs` likewise).

## Behaviour
- Time grid: step `s` (0-based) has telemetry events at `BaseNs + s*StepIntervalNs` and, per rank, one AllReduce
  `NCCLCollectiveEvent` at `BaseNs + s*StepIntervalNs + StepIntervalNs/2`. All ranks share the same timestamp within a step.
  Tests recover the step as `(ts - BaseNs) / StepIntervalNs`.
- Per rank per step exactly 4 events: `GPUMetricEvent`, `ThermalEvent`, `NVLinkEvent` (LinkID 0), `NCCLCollectiveEvent`
  (OpType "AllReduce"). Total `Ranks*Steps*4`. `Events` sorted by non-decreasing timestamp (stable, so ties keep rank order).
- Healthy values: collective `DurationNs = BaseDurationNs*(1+n)`, NVLink `ThroughputGB = NominalNVLinkGBs*(1+n)`, where `n` is
  Gaussian with the jitter level's sigma, clamped to +/-3 sigma (so healthy durations stay under 1.5x base and throughput
  over 0.7x nominal even at high jitter). Temperature uniform in [60,70], `MemBandwidth` uniform in [40,60], ECC counters 0,
  `ThrottleActive` false.
- Fault window: `FaultRank` during steps `FaultStartStep .. FaultStartStep+FaultSteps-1` inclusive. Inside it:
  collective duration is multiplied by 3.0 for every fault type; plus the signature:
  - thermal_throttle: `ThrottleActive` true, temperature uniform in [92,95]
  - ecc_errors: `ECCErrorsSBE` increases by at least 1 each fault step (cumulative, never decreases, 0 before the window)
  - nvlink_degraded: throughput multiplied by 0.3
  - memory_pressure: `MemBandwidth` uniform in [96,100]
  - host_stall: no hardware signature at all (only the slowdown); its ExpectedCause is "unknown"
- `ExpectedCause` strings (from the catalogue, to line up with `attribution.CauseType` values without importing it):
  thermal_throttle, ecc_errors, nvlink_degraded, memory_pressure, and "unknown" for host_stall. Control: empty string.
- `Generate` returns an error for: Ranks < 2, Steps < 1, unknown fault or jitter, and (for a fault) FaultRank >= Ranks,
  FaultSteps < 1, FaultStartStep < 0, or the window extending past Steps. `FaultNone` ignores the fault fields.
- Determinism: one `rand.New(rand.NewSource(cfg.Seed))` consumed in a fixed order (step, then rank, then event); no map
  iteration, no wall clock. Same Config -> `reflect.DeepEqual` Scenarios.
- The values above are nominal simulation parameters, not measurements of any hardware. Document each fault in the
  `Catalogue()` Description/Signature strings and in a package comment that says the data is simulated.

## Verify before opening the PR
Run the four acceptance commands in order. The second fails while the build-tag line is still present, so remove that
line (and the blank line after it) only after the first command passes. Then `gofmt -l internal/sim` must print nothing.

## Log
- 2026-10-30 draft
