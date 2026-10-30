//go:build spec_gfr_eval

package eval_test

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/ryanhamby/gpu-flight-recorder/internal/correlator"
	"github.com/ryanhamby/gpu-flight-recorder/internal/eval"
	"github.com/ryanhamby/gpu-flight-recorder/internal/sim"
)

const ranks = 6

func opts(p eval.Pipeline) eval.Options {
	o := eval.DefaultOptions()
	o.Seeds = []int64{1, 2}
	o.Ranks, o.Steps = ranks, 20
	o.Pipeline = p
	return o
}

// truthRank is where the simulator's injected fault lives in eval's scenarios.
// The harness must take it from Scenario.Truth, so fakes read it from there.
func anomaly(rank uint32) correlator.Anomaly {
	return correlator.Anomaly{StragglerRank: rank, OpType: "AllReduce"}
}

// perfect reports exactly the injected fault and its expected cause, nothing for the control.
func perfect(sc sim.Scenario, _ eval.Options) (eval.Observation, error) {
	if !sc.Truth.HasFault {
		return eval.Observation{}, nil
	}
	return eval.Observation{Anomalies: []correlator.Anomaly{anomaly(sc.Truth.Rank)}, Cause: sc.Truth.ExpectedCause}, nil
}

func run(t *testing.T, p eval.Pipeline) eval.Report {
	t.Helper()
	r, err := eval.Run(opts(p))
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	return r
}

func TestCaseMatrixAndLabels(t *testing.T) {
	r := run(t, perfect)
	// (none + 5 faults) x 3 jitter levels x 2 seeds
	if len(r.Cases) != 6*3*2 {
		t.Fatalf("got %d cases, want 36", len(r.Cases))
	}
	if r.Data != "simulated" || r.SchemaVersion != 1 {
		t.Fatalf("Data=%q SchemaVersion=%d, want \"simulated\" and 1", r.Data, r.SchemaVersion)
	}
	wantFaults := []string{"none"}
	for _, f := range sim.Catalogue() {
		wantFaults = append(wantFaults, string(f.Type))
	}
	i := 0
	for _, f := range wantFaults {
		for _, j := range []string{"low", "medium", "high"} {
			for _, s := range []int64{1, 2} {
				c := r.Cases[i]
				i++
				if c.Fault != f || c.Jitter != j || c.Seed != s {
					t.Fatalf("case %d = (%s,%s,%d), want (%s,%s,%d): order is fault (none first, then catalogue order), jitter, seed", i-1, c.Fault, c.Jitter, c.Seed, f, j, s)
				}
				if f == "none" {
					if c.GroundTruthRank != nil || c.GroundTruthCause != "" {
						t.Errorf("control case must have null ground_truth_rank and empty cause: %+v", c)
					}
				} else if c.GroundTruthRank == nil || c.GroundTruthCause == "" {
					t.Errorf("fault case %s lacks ground truth: %+v", f, c)
				}
			}
		}
	}
}

func TestPerfectPipelineScores(t *testing.T) {
	r := run(t, perfect)
	for _, c := range r.Cases {
		if !c.DetectionCorrect || c.FalseStragglerCount != 0 {
			t.Errorf("%s: perfect pipeline should be detection_correct with 0 false stragglers: %+v", c.Fault, c)
		}
		if c.Fault != "none" {
			if c.DetectedRank == nil || *c.DetectedRank != *c.GroundTruthRank {
				t.Errorf("%s: detected rank not reported: %+v", c.Fault, c)
			}
			if !c.AttributionCorrect || c.AttributedCause != c.GroundTruthCause {
				t.Errorf("%s: attribution should match truth: %+v", c.Fault, c)
			}
		} else if c.DetectedRank != nil || c.AttributedCause != "none" {
			t.Errorf("control: want null detected_rank and attributed_cause \"none\": %+v", c)
		}
	}
	s := r.Summary
	if s.Cases != 36 || s.FaultCases != 30 || s.ControlCases != 6 || s.DetectionCorrect != 36 || s.AttributionCorrect != 30 || s.ControlFalseStragglerTotal != 0 {
		t.Errorf("bad summary %+v", s)
	}
}

func TestFalseStragglersAndWrongRank(t *testing.T) {
	noisy := func(sc sim.Scenario, _ eval.Options) (eval.Observation, error) {
		if !sc.Truth.HasFault { // control: two spurious anomalies on rank 0
			return eval.Observation{Anomalies: []correlator.Anomaly{anomaly(0), anomaly(0)}}, nil
		}
		wrong := (sc.Truth.Rank + 1) % ranks
		// the wrong rank is flagged twice, the true rank once: majority picks the wrong rank
		return eval.Observation{Anomalies: []correlator.Anomaly{anomaly(wrong), anomaly(sc.Truth.Rank), anomaly(wrong)}, Cause: "unknown"}, nil
	}
	r := run(t, noisy)
	for _, c := range r.Cases {
		if c.Fault == "none" {
			if c.DetectedRank == nil || *c.DetectedRank != 0 || c.DetectionCorrect || c.FalseStragglerCount != 2 {
				t.Errorf("control with spurious anomalies mis-scored: %+v", c)
			}
			continue
		}
		if c.DetectionCorrect || c.FalseStragglerCount != 2 || c.DetectedRank == nil || *c.DetectedRank == *c.GroundTruthRank {
			t.Errorf("%s: wrong-rank majority mis-scored: %+v", c.Fault, c)
		}
		// "unknown" is the correct cause only for host_stall
		if want := c.Fault == "host_stall"; c.AttributionCorrect != want {
			t.Errorf("%s: attribution_correct = %v, want %v", c.Fault, c.AttributionCorrect, want)
		}
	}
	if r.Summary.ControlFalseStragglerTotal != 2*6 || r.Summary.DetectionCorrect != 0 {
		t.Errorf("bad summary %+v", r.Summary)
	}
}

func TestTieBreaksToLowestRank(t *testing.T) {
	p := func(sc sim.Scenario, _ eval.Options) (eval.Observation, error) {
		return eval.Observation{Anomalies: []correlator.Anomaly{anomaly(4), anomaly(2)}}, nil
	}
	for _, c := range run(t, p).Cases {
		if c.DetectedRank == nil || *c.DetectedRank != 2 {
			t.Fatalf("tie between ranks 4 and 2 must pick 2: %+v", c)
		}
	}
}

func TestJSONShapeAndDeterminism(t *testing.T) {
	var a, b bytes.Buffer
	r := run(t, perfect)
	if err := eval.WriteJSON(&a, r); err != nil {
		t.Fatal(err)
	}
	if err := eval.WriteJSON(&b, run(t, perfect)); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Fatal("same options must produce byte-identical JSON (no timestamps, hostnames or randomness)")
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(a.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	keys(t, "report", doc, "schema_version", "data", "cases", "summary")
	var data string
	_ = json.Unmarshal(doc["data"], &data)
	if data != "simulated" {
		t.Errorf("data = %q, want \"simulated\"", data)
	}
	var cases []map[string]json.RawMessage
	if err := json.Unmarshal(doc["cases"], &cases); err != nil {
		t.Fatal(err)
	}
	keys(t, "case", cases[0], "seed", "fault", "jitter", "ground_truth_rank", "ground_truth_cause", "detected_rank",
		"attributed_cause", "false_straggler_count", "detection_correct", "attribution_correct")
	var summary map[string]json.RawMessage
	_ = json.Unmarshal(doc["summary"], &summary)
	// counts only: no rates, percentages or "accuracy" fields may be added without a captured run behind them
	keys(t, "summary", summary, "cases", "fault_cases", "control_cases", "detection_correct", "attribution_correct", "control_false_straggler_total")
	if string(cases[0]["ground_truth_rank"]) != "null" || string(cases[0]["detected_rank"]) != "null" {
		t.Errorf("control row must serialise absent ranks as null, got %s / %s", cases[0]["ground_truth_rank"], cases[0]["detected_rank"])
	}
}

func keys(t *testing.T, what string, m map[string]json.RawMessage, want ...string) {
	t.Helper()
	if len(m) != len(want) {
		t.Errorf("%s has %d keys, want exactly %v", what, len(m), want)
	}
	for _, k := range want {
		if _, ok := m[k]; !ok {
			t.Errorf("%s missing key %q", what, k)
		}
	}
}

func TestOptionsValidationAndPipelineErrors(t *testing.T) {
	o := opts(perfect)
	o.Seeds = nil
	if _, err := eval.Run(o); err == nil {
		t.Error("empty Seeds must be an error")
	}
	o = opts(perfect)
	o.Ranks = 1
	if _, err := eval.Run(o); err == nil {
		t.Error("Ranks < 2 must be an error")
	}
	failing := func(sim.Scenario, eval.Options) (eval.Observation, error) {
		return eval.Observation{}, errBoom
	}
	if _, err := eval.Run(opts(failing)); err == nil {
		t.Error("pipeline error must propagate")
	}
}

type boom struct{}

func (boom) Error() string { return "boom" }

var errBoom error = boom{}

// The default pipeline drives the real correlator and attribution engine. Their
// detection logic is still being written, so this test only requires that the
// end-to-end run completes and yields well-formed rows (a detector that finds
// nothing is reported honestly as null/none, never as a success).
func TestDefaultPipelineRunsEndToEnd(t *testing.T) {
	o := eval.DefaultOptions()
	o.Seeds, o.Ranks, o.Steps = []int64{1}, ranks, 20
	if o.Pipeline != nil {
		t.Fatal("DefaultOptions().Pipeline must be nil, meaning eval.DefaultPipeline")
	}
	r, err := eval.Run(o)
	if err != nil {
		t.Fatalf("Run with default pipeline: %v", err)
	}
	if len(r.Cases) != 18 || r.Data != "simulated" {
		t.Fatalf("got %d cases, data %q", len(r.Cases), r.Data)
	}
	for _, c := range r.Cases {
		if c.AttributedCause == "" {
			t.Errorf("attributed_cause must be \"none\" rather than empty: %+v", c)
		}
		if c.DetectedRank == nil && (c.DetectionCorrect != (c.Fault == "none")) {
			t.Errorf("no detection is correct only for the control: %+v", c)
		}
	}
	if _, err := eval.DefaultPipeline(sim.Scenario{}, o); err != nil {
		t.Fatalf("DefaultPipeline on an empty scenario: %v", err)
	}
}
