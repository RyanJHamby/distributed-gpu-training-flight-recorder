---
id: gfr-eval
goal: An evaluation harness runs every simulated fault (plus a no-fault control) at three jitter levels through the real correlator and attribution engine and writes one results JSON that is labelled data "simulated" and records ground truth, detected rank, attribution and false-straggler counts per case.
acceptance:
  - go test -tags spec_gfr_eval ./internal/eval/... -count=1
  - sh -c 'test -z "$(grep -l spec_gfr_eval internal/eval/eval_test.go)"'
  - go test ./internal/eval/... -count=1
  - go vet ./internal/eval/... ./cmd/gfr-eval/...
  - go build -o /dev/null ./cmd/gfr-eval
scope:
  - internal/eval/eval.go
  - internal/eval/eval_test.go
  - cmd/gfr-eval/main.go
non_goals:
  - depends on spec gfr-sim being merged first (internal/sim must exist); do not recreate or modify internal/sim
  - do not change any assertion in internal/eval/eval_test.go; the only permitted edit is deleting its first line (`//go:build spec_gfr_eval`) and the blank line after it, which makes the third command run the test in normal CI
  - no changes to internal/correlator, internal/attribution, internal/buffer or internal/types; the detector is still mostly TODO and this harness must report that honestly (null detected_rank, attributed_cause "none"), never paper over it
  - do not commit any results file, and do not quote a detection rate, accuracy or any other number in README, docs or code comments; only counts appear in the JSON
  - no new third-party dependencies (standard library plus existing internal packages), no GPU, NVML, network or wall-clock use; do not touch cmd/gfr (it currently fails to build because go.sum lacks cobra)
  - do not import internal/transport or google.golang.org/grpc (go.sum is incomplete in this repo today)
risk_tier: 2
max_runs: 3
autonomy: inherit
status: draft
---

## Why
The site's earlier "30/30 detection" claim was withdrawn because no simulator or captured result existed. The owner's rule:
quote no number unless it comes from a captured run (`pow run NAME -- CMD` saves command, output and hardware to
`results/`), name the hardware, and label simulated data as simulated. This harness is the measuring instrument. It produces
data only; any later claim must point at a `pow run` capture of it. Today the correlator's `IngestEvent`/`DetectAnomalies`
and `ComputeStats` are TODO stubs, so the honest first result is "nothing detected"; that is a valid, useful baseline.

## Where
- `internal/eval/eval.go` (package `eval`), `cmd/gfr-eval/main.go` (thin main: flag `-out PATH`, default stdout; calls
  `eval.Run(eval.DefaultOptions())` then `eval.WriteJSON`; a usage comment shows the capture command
  `pow run gfr-eval-sim -- go run ./cmd/gfr-eval`).
- Uses `internal/sim` from spec `gfr-sim` (types `sim.Scenario`, `sim.Config`, `sim.Catalogue()`, `sim.DataLabel`,
  `Scenario.Truth`), `internal/correlator` (`NewCorrelator`, `IngestEvent`, `DetectAnomalies`, `EventsForRank`, `Anomaly`),
  `internal/attribution` (`NewEngine`, `Attribute`, `CausalChain`) and `types.DefaultCoordinatorConfig()`.
- The test `internal/eval/eval_test.go` (build tag `spec_gfr_eval`) is the contract. Read it first.

## API (exact names the test uses)
```go
type Observation struct { Anomalies []correlator.Anomaly; Cause string } // Cause "" = no attribution
type Pipeline func(sim.Scenario, Options) (Observation, error)
type Options struct { Seeds []int64; Ranks, Steps int; Pipeline Pipeline } // nil Pipeline = DefaultPipeline
func DefaultOptions() Options  // Seeds {1,2,3}, Ranks 8, Steps 50, Pipeline nil
func DefaultPipeline(sim.Scenario, Options) (Observation, error)
func Run(Options) (Report, error)
func WriteJSON(io.Writer, Report) error
type CaseResult struct { Seed int64; Fault, Jitter string; GroundTruthRank *uint32; GroundTruthCause string; DetectedRank *uint32;
  AttributedCause string; FalseStragglerCount int; DetectionCorrect, AttributionCorrect bool }
type Summary struct { Cases, FaultCases, ControlCases, DetectionCorrect, AttributionCorrect, ControlFalseStragglerTotal int }
type Report struct { SchemaVersion int; Data string; Cases []CaseResult; Summary Summary }
```
JSON keys (snake_case, exactly these, no others): report `schema_version` (1), `data` ("simulated"), `cases`, `summary`;
case `seed, fault, jitter, ground_truth_rank, ground_truth_cause, detected_rank, attributed_cause, false_straggler_count,
detection_correct, attribution_correct`; summary `cases, fault_cases, control_cases, detection_correct, attribution_correct,
control_false_straggler_total`. The two rank fields are `null` when absent (pointers, no `omitempty`). Output must be
byte-identical for identical Options: no timestamps, hostnames or map-order dependence (hardware is recorded by `pow run`, not here).

## Behaviour
- Cases iterate: fault (`none` first, then `sim.Catalogue()` order) x jitter (low, medium, high) x `Options.Seeds` order.
  Scenario config: `Ranks`, `Steps` from Options; for a fault, `FaultRank` 2, `FaultStartStep` Steps/4, `FaultSteps` Steps/2.
- `Run` errors on empty Seeds, Ranks < 2, Steps < 1, and propagates `Generate` and pipeline errors.
- `DefaultPipeline`: new `correlator.NewCorrelator(cfg.AnomalyThresholdSigma)` using `types.DefaultCoordinatorConfig()`;
  `IngestEvent(e.Rank(), e)` for every scenario event in order; `DetectAnomalies(cfg.LookbackWindow)`; if any anomaly,
  `attribution.NewEngine(int64(cfg.AttributionWindow)).Attribute(anomalies[0], c.EventsForRank(rank, 0))` and
  `Cause` = first `CausalChain` entry's cause (empty if the chain is empty).
- Scoring from the Observation: `detected_rank` = the rank with the most anomalies, ties to the lowest rank, null if no
  anomalies. `false_straggler_count` = anomalies whose `StragglerRank` is not the injected fault rank (every anomaly in the
  control). `attributed_cause` = Cause, or the string "none". Fault case: `detection_correct` = detected == truth rank,
  `attribution_correct` = attributed_cause == `Truth.ExpectedCause` (host_stall expects "unknown"). Control:
  `detection_correct` = no detection; `attribution_correct` false. Summary fields are plain counts.
- The report `data` is `sim.DataLabel`. Do not add rates, percentages or an "accuracy" field.

## Verify
Run the five acceptance commands in order. The second fails while the build-tag line is present, so delete the tag line
(plus the blank line) only after the first passes. `gofmt -l internal cmd/gfr-eval` must print nothing. Note
`go build ./...` and `go test ./...` for the whole module are red in this repo before this spec (missing go.sum entries for
cobra and grpc, affecting cmd/gfr and internal/transport only); do not try to fix that here.

## Log
- 2026-10-30 draft
