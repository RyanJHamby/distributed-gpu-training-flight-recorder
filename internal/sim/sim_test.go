//go:build spec_gfr_sim

package sim_test

import (
	"reflect"
	"sort"
	"testing"

	"github.com/ryanhamby/gpu-flight-recorder/internal/sim"
	"github.com/ryanhamby/gpu-flight-recorder/internal/types"
)

var (
	allFaults  = []sim.FaultType{sim.FaultThermalThrottle, sim.FaultECCErrors, sim.FaultNVLinkDegraded, sim.FaultMemoryPressure, sim.FaultHostStall}
	allJitters = []sim.JitterLevel{sim.JitterLow, sim.JitterMedium, sim.JitterHigh}
	seeds      = []int64{1, 2, 42}
)

func cfg(seed int64, f sim.FaultType, j sim.JitterLevel) sim.Config {
	c := sim.Config{Seed: seed, Ranks: 8, Steps: 40, Fault: f, Jitter: j}
	if f != sim.FaultNone {
		c.FaultRank, c.FaultStartStep, c.FaultSteps = 5, 10, 20 // steps 10..29 inclusive
	}
	return c
}

func step(ts int64) int { return int((ts - sim.BaseNs) / sim.StepIntervalNs) }

func gen(t *testing.T, c sim.Config) sim.Scenario {
	t.Helper()
	sc, err := sim.Generate(c)
	if err != nil {
		t.Fatalf("Generate(%+v): %v", c, err)
	}
	return sc
}

func TestCatalogue(t *testing.T) {
	cat := sim.Catalogue()
	if len(cat) != len(allFaults) {
		t.Fatalf("catalogue has %d entries, want %d", len(cat), len(allFaults))
	}
	names := make([]string, len(cat))
	seen := map[sim.FaultType]bool{}
	for i, s := range cat {
		names[i] = string(s.Type)
		seen[s.Type] = true
		if s.Description == "" || s.ExpectedCause == "" || s.Signature == "" {
			t.Errorf("%s: Description, ExpectedCause and Signature must all be non-empty", s.Type)
		}
	}
	if !sort.StringsAreSorted(names) {
		t.Errorf("catalogue must be sorted by Type, got %v", names)
	}
	for _, f := range allFaults {
		if !seen[f] {
			t.Errorf("fault %q missing from catalogue", f)
		}
	}
	if sim.DataLabel != "simulated" {
		t.Errorf("DataLabel = %q, want \"simulated\"", sim.DataLabel)
	}
}

func TestSeedDeterminism(t *testing.T) {
	c := cfg(7, sim.FaultThermalThrottle, sim.JitterMedium)
	a, b := gen(t, c), gen(t, c)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("same Config produced different Scenarios")
	}
	if a.Data != "simulated" {
		t.Errorf("Scenario.Data = %q, want \"simulated\"", a.Data)
	}
	c2 := c
	c2.Seed = 8
	if reflect.DeepEqual(a.Events, gen(t, c2).Events) {
		t.Fatal("different seeds produced identical events")
	}
}

func TestShapeAndOrdering(t *testing.T) {
	c := cfg(3, sim.FaultECCErrors, sim.JitterLow)
	sc := gen(t, c)
	if want := c.Ranks * c.Steps * 4; len(sc.Events) != want {
		t.Fatalf("got %d events, want ranks*steps*4 = %d", len(sc.Events), want)
	}
	var last int64 = -1
	collectiveTS := map[int]int64{}
	perStepRank := map[[2]int]int{}
	for _, e := range sc.Events {
		if e.TimestampNs() < last {
			t.Fatal("events must be sorted by non-decreasing timestamp")
		}
		last = e.TimestampNs()
		if int(e.Rank()) >= c.Ranks {
			t.Fatalf("rank %d out of range", e.Rank())
		}
		if n, ok := e.(*types.NCCLCollectiveEvent); ok {
			s := step(n.Timestamp)
			if prev, seen := collectiveTS[s]; seen && prev != n.Timestamp {
				t.Errorf("step %d: collectives of different ranks must share one timestamp (no clock skew)", s)
			}
			collectiveTS[s] = n.Timestamp
			perStepRank[[2]int{s, int(n.Rank())}]++
			if n.OpType != "AllReduce" || n.DurationNs <= 0 {
				t.Errorf("bad collective %+v", n)
			}
		}
	}
	if len(perStepRank) != c.Ranks*c.Steps {
		t.Errorf("want exactly one collective per rank per step, got %d distinct (step,rank) pairs", len(perStepRank))
	}
}

func TestControlHasNoStraggler(t *testing.T) {
	for _, j := range allJitters {
		for _, s := range seeds {
			sc := gen(t, cfg(s, sim.FaultNone, j))
			if sc.Truth.HasFault {
				t.Fatal("control must have Truth.HasFault == false")
			}
			for _, e := range sc.Events {
				if n, ok := e.(*types.NCCLCollectiveEvent); ok && float64(n.DurationNs) > 1.5*sim.BaseDurationNs {
					t.Fatalf("control (jitter %s, seed %d): duration %d exceeds 1.5x base", j, s, n.DurationNs)
				}
			}
		}
	}
}

func TestFaultTruthAndSlowdown(t *testing.T) {
	for _, f := range allFaults {
		for _, j := range allJitters {
			for _, s := range seeds {
				c := cfg(s, f, j)
				sc := gen(t, c)
				tr := sc.Truth
				if !tr.HasFault || tr.Fault != f || tr.Rank != 5 || tr.FirstStep != 10 || tr.LastStep != 29 {
					t.Fatalf("%s/%s/%d: bad truth %+v", f, j, s, tr)
				}
				var inSum float64
				var inN int
				for _, e := range sc.Events {
					n, ok := e.(*types.NCCLCollectiveEvent)
					if !ok {
						continue
					}
					st := step(n.Timestamp)
					inFault := n.Rank() == tr.Rank && st >= tr.FirstStep && st <= tr.LastStep
					if inFault {
						inSum += float64(n.DurationNs)
						inN++
					} else if float64(n.DurationNs) > 1.5*sim.BaseDurationNs {
						t.Fatalf("%s/%s/%d: healthy collective rank %d step %d is slow (%d ns)", f, j, s, n.Rank(), st, n.DurationNs)
					}
				}
				if inN != 20 {
					t.Fatalf("%s: %d collectives in fault window, want 20", f, inN)
				}
				if mean := inSum / float64(inN); mean < 2.0*sim.BaseDurationNs {
					t.Errorf("%s/%s/%d: fault mean %.0f ns is below 2x base", f, j, s, mean)
				}
			}
		}
	}
}

// Each fault leaves its documented hardware signature on the fault rank inside
// the fault window, and never anywhere else. host_stall leaves none.
func TestSignatures(t *testing.T) {
	for _, f := range allFaults {
		for _, j := range allJitters {
			for _, s := range seeds {
				sc := gen(t, cfg(s, f, j))
				tr := sc.Truth
				var sbeBefore, sbeEnd uint64
				for _, e := range sc.Events {
					st := step(e.TimestampNs())
					in := e.Rank() == tr.Rank && st >= tr.FirstStep && st <= tr.LastStep
					tag := string(f) + "/" + string(j)
					switch v := e.(type) {
					case *types.ThermalEvent:
						if throttle := v.ThrottleActive; throttle != (in && f == sim.FaultThermalThrottle) {
							t.Fatalf("%s: ThrottleActive=%v rank %d step %d", tag, throttle, v.Rank(), st)
						}
						if in && f == sim.FaultThermalThrottle && v.TemperatureC < 90 {
							t.Fatalf("%s: throttled temperature %.1f < 90", tag, v.TemperatureC)
						}
						if !(in && f == sim.FaultThermalThrottle) && v.TemperatureC > 75 {
							t.Fatalf("%s: healthy temperature %.1f > 75", tag, v.TemperatureC)
						}
					case *types.NVLinkEvent:
						degraded := in && f == sim.FaultNVLinkDegraded
						if degraded && v.ThroughputGB > 0.5*sim.NominalNVLinkGBs {
							t.Fatalf("%s: degraded throughput %.1f > 0.5 nominal", tag, v.ThroughputGB)
						}
						if !degraded && v.ThroughputGB < 0.7*sim.NominalNVLinkGBs {
							t.Fatalf("%s: healthy throughput %.1f < 0.7 nominal", tag, v.ThroughputGB)
						}
					case *types.GPUMetricEvent:
						pressure := in && f == sim.FaultMemoryPressure
						if pressure != (v.MemBandwidth >= 95) {
							t.Fatalf("%s: MemBandwidth %.1f, pressure=%v rank %d step %d", tag, v.MemBandwidth, pressure, v.Rank(), st)
						}
						if v.Rank() != tr.Rank || f != sim.FaultECCErrors {
							if v.ECCErrorsSBE != 0 || v.ECCErrorsDBE != 0 {
								t.Fatalf("%s: unexpected ECC errors on rank %d", tag, v.Rank())
							}
						} else {
							if st < tr.FirstStep && v.ECCErrorsSBE != 0 {
								t.Fatalf("%s: ECC errors before fault start", tag)
							}
							if st == tr.FirstStep-1 {
								sbeBefore = v.ECCErrorsSBE
							}
							if st == tr.LastStep {
								sbeEnd = v.ECCErrorsSBE
							}
						}
					}
				}
				if f == sim.FaultECCErrors && sbeEnd < sbeBefore+uint64(tr.LastStep-tr.FirstStep+1) {
					t.Errorf("%s: SBE counter grew %d -> %d, want at least +1 per fault step", f, sbeBefore, sbeEnd)
				}
			}
		}
	}
}

func TestSBECounterMonotonic(t *testing.T) {
	sc := gen(t, cfg(9, sim.FaultECCErrors, sim.JitterHigh))
	var prev uint64
	for _, e := range sc.Events {
		if m, ok := e.(*types.GPUMetricEvent); ok && m.Rank() == sc.Truth.Rank {
			if m.ECCErrorsSBE < prev {
				t.Fatal("SBE counter must never decrease")
			}
			prev = m.ECCErrorsSBE
		}
	}
}

func TestExpectedCauseMatchesCatalogue(t *testing.T) {
	want := map[sim.FaultType]string{}
	for _, s := range sim.Catalogue() {
		want[s.Type] = s.ExpectedCause
	}
	for _, f := range allFaults {
		if got := gen(t, cfg(1, f, sim.JitterLow)).Truth.ExpectedCause; got != want[f] {
			t.Errorf("%s: Truth.ExpectedCause %q != catalogue %q", f, got, want[f])
		}
	}
	if got := gen(t, cfg(1, sim.FaultNone, sim.JitterLow)).Truth.ExpectedCause; got != "" {
		t.Errorf("control ExpectedCause = %q, want empty", got)
	}
}

func TestValidation(t *testing.T) {
	bad := map[string]sim.Config{
		"one rank":         {Seed: 1, Ranks: 1, Steps: 10, Fault: sim.FaultNone, Jitter: sim.JitterLow},
		"zero steps":       {Seed: 1, Ranks: 4, Steps: 0, Fault: sim.FaultNone, Jitter: sim.JitterLow},
		"unknown fault":    {Seed: 1, Ranks: 4, Steps: 10, Fault: "gremlins", Jitter: sim.JitterLow},
		"unknown jitter":   {Seed: 1, Ranks: 4, Steps: 10, Fault: sim.FaultNone, Jitter: "extreme"},
		"fault rank oob":   {Seed: 1, Ranks: 4, Steps: 10, Fault: sim.FaultHostStall, Jitter: sim.JitterLow, FaultRank: 4, FaultStartStep: 1, FaultSteps: 2},
		"window past end":  {Seed: 1, Ranks: 4, Steps: 10, Fault: sim.FaultHostStall, Jitter: sim.JitterLow, FaultRank: 1, FaultStartStep: 8, FaultSteps: 5},
		"empty fault span": {Seed: 1, Ranks: 4, Steps: 10, Fault: sim.FaultHostStall, Jitter: sim.JitterLow, FaultRank: 1, FaultStartStep: 2, FaultSteps: 0},
	}
	for name, c := range bad {
		if _, err := sim.Generate(c); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}
