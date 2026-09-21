package attribution

import (
	"testing"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/correlator"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

const sec = int64(1e9)

var anom = correlator.Anomaly{StragglerRank: 3, OpType: "AllReduce", Hits: 40, Groups: 50,
	LagNs: 3e6, StartNs: 100 * sec, EndNs: 110 * sec}

func gpu(ts int64, mut func(*types.GPUMetricEvent)) types.Event {
	m := types.GPUMetricEvent{BaseEvent: types.BaseEvent{Timestamp: ts, RankID: 3}, Utilization: 95, MemBandwidth: 50}
	if mut != nil {
		mut(&m)
	}
	return m
}

// healthy baseline before the window, then per-case events during it.
func series(mut func(*types.GPUMetricEvent)) []types.Event {
	return []types.Event{gpu(50*sec, nil), gpu(90*sec, nil), gpu(101*sec, mut), gpu(105*sec, mut), gpu(109*sec, mut)}
}

func top(t *testing.T, a Attribution) CauseType {
	t.Helper()
	if len(a.CausalChain) == 0 {
		t.Fatal("empty chain")
	}
	return a.CausalChain[0].Cause
}

func TestAttributeCauses(t *testing.T) {
	e := NewEngine(5 * sec)
	pcie := func(ts int64, mbs float64) types.Event {
		return types.PCIeBandwidthEvent{BaseEvent: types.BaseEvent{Timestamp: ts}, ReadBandwidthMBs: mbs}
	}
	nvl := func(ts int64, g float64) types.Event {
		return types.NVLinkEvent{BaseEvent: types.BaseEvent{Timestamp: ts}, ThroughputGB: g}
	}
	cases := []struct {
		name string
		evs  []types.Event
		want CauseType
	}{
		{"thermal", series(func(m *types.GPUMetricEvent) { m.ThrottleReasons = throttleHWThermal }), CauseThermalThrottle},
		{"thermal via ThermalEvent", []types.Event{types.ThermalEvent{BaseEvent: types.BaseEvent{Timestamp: 105 * sec}, ThrottleActive: true}}, CauseThermalThrottle},
		{"power cap", series(func(m *types.GPUMetricEvent) { m.ThrottleReasons = throttleSWPowerCap }), CausePowerThrottle},
		{"clock lock, no throttle bit", series2(func(m *types.GPUMetricEvent) { m.SMClockMHz = 900 }), CauseClockReduced},
		{"ecc dbe", series(func(m *types.GPUMetricEvent) { m.ECCErrorsDBE = 2 }), CauseECCErrors},
		{"pcie drop", []types.Event{pcie(50*sec, 12000), pcie(90*sec, 12000), pcie(103*sec, 3000), pcie(106*sec, 3000)}, CausePCIeBandwidth},
		{"nvlink drop", []types.Event{nvl(50*sec, 200), nvl(90*sec, 200), nvl(103*sec, 40)}, CauseNVLinkDegraded},
		{"memory pressure", series(func(m *types.GPUMetricEvent) { m.MemBandwidth = 97 }), CauseMemoryPressure},
		{"host stall", series(func(m *types.GPUMetricEvent) { m.Utilization = 8 }), CauseHostStall},
		{"nothing to go on", nil, CauseUnknown},
		{"healthy signals", series(nil), CauseUnknown},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := top(t, e.Attribute(anom, c.evs)); got != c.want {
				t.Fatalf("got %s want %s", got, c.want)
			}
		})
	}
}

func TestBandwidthRuleNeedsBaseline(t *testing.T) {
	e := NewEngine(5 * sec)
	a := e.Attribute(anom, []types.Event{types.PCIeBandwidthEvent{BaseEvent: types.BaseEvent{Timestamp: 105 * sec}, ReadBandwidthMBs: 1}})
	if top(t, a) != CauseUnknown {
		t.Fatalf("fired without a baseline: %+v", a.CausalChain)
	}
}

func TestIdleGPUThatWasThrottledIsNotHostStall(t *testing.T) {
	e := NewEngine(5 * sec)
	a := e.Attribute(anom, series(func(m *types.GPUMetricEvent) { m.Utilization = 5; m.ThrottleReasons = throttleSWThermal }))
	for _, l := range a.CausalChain {
		if l.Cause == CauseHostStall {
			t.Fatal("host_stall should be suppressed under throttle")
		}
	}
	if top(t, a) != CauseThermalThrottle {
		t.Fatalf("got %s", top(t, a))
	}
}

func TestMultipleCausesRankedByConfidence(t *testing.T) {
	e := NewEngine(5 * sec)
	a := e.Attribute(anom, series(func(m *types.GPUMetricEvent) { m.ThrottleReasons = throttleHWThermal; m.ECCErrorsDBE = 1 }))
	if len(a.CausalChain) != 2 || a.CausalChain[0].Confidence < a.CausalChain[1].Confidence {
		t.Fatalf("want two causes, highest confidence first: %+v", a.CausalChain)
	}
	if top(t, a) != CauseThermalThrottle || a.CausalChain[1].Cause != CauseECCErrors {
		t.Fatalf("order: %s, %s", top(t, a), a.CausalChain[1].Cause)
	}
}

func TestRuledOutOnlyWhenDataExists(t *testing.T) {
	e := NewEngine(5 * sec)
	a := e.Attribute(anom, series(func(m *types.GPUMetricEvent) { m.ThrottleReasons = throttleHWThermal }))
	has := func(c CauseType) bool {
		for _, r := range a.RuledOut {
			if r == c {
				return true
			}
		}
		return false
	}
	if !has(CauseECCErrors) || has(CausePCIeBandwidth) {
		t.Fatalf("ruled out: %v", a.RuledOut)
	}
}

func TestEventsAfterWindowIgnored(t *testing.T) {
	e := NewEngine(5 * sec)
	a := e.Attribute(anom, []types.Event{gpu(200*sec, func(m *types.GPUMetricEvent) { m.ThrottleReasons = throttleHWThermal })})
	if top(t, a) != CauseUnknown {
		t.Fatal("used an event outside the window")
	}
}

func TestSummaryMentionsRankAndCause(t *testing.T) {
	a := NewEngine(5*sec).Attribute(anom, series(func(m *types.GPUMetricEvent) { m.ThrottleReasons = throttleHWThermal }))
	if a.Summary == "" || a.Anomaly.StragglerRank != 3 {
		t.Fatal(a.Summary)
	}
}

// series2 is series with an SM clock baseline, for clock-rule cases.
func series2(mut func(*types.GPUMetricEvent)) []types.Event {
	base := func(ts int64) types.Event { return gpu(ts, func(m *types.GPUMetricEvent) { m.SMClockMHz = 1900 }) }
	during := func(ts int64) types.Event {
		return gpu(ts, func(m *types.GPUMetricEvent) { m.SMClockMHz = 1900; mut(m) })
	}
	return []types.Event{base(50 * sec), base(90 * sec), during(101 * sec), during(105 * sec), during(109 * sec)}
}

func TestClockRuleNeedsBaselineAndStaysBelowThrottle(t *testing.T) {
	e := NewEngine(5 * sec)
	only := []types.Event{gpu(105*sec, func(m *types.GPUMetricEvent) { m.SMClockMHz = 300 })}
	if top(t, e.Attribute(anom, only)) != CauseUnknown {
		t.Fatal("clock rule fired without a baseline")
	}
	a := e.Attribute(anom, series2(func(m *types.GPUMetricEvent) { m.SMClockMHz = 900; m.ThrottleReasons = throttleHWThermal }))
	if top(t, a) != CauseThermalThrottle {
		t.Fatalf("explicit throttle should outrank clock drop: %+v", a.CausalChain)
	}
}

func procs(n int) func(*types.GPUMetricEvent) {
	return func(m *types.GPUMetricEvent) { m.ComputeProcs = n }
}

func TestContentionRule(t *testing.T) {
	e := NewEngine(5 * sec)
	mk := func(baseline, during int) []types.Event {
		g := func(ts int64, n int) types.Event { return gpu(ts, procs(n)) }
		return []types.Event{g(50*sec, baseline), g(90*sec, baseline), g(101*sec, during), g(105*sec, during), g(109*sec, during)}
	}
	if got := top(t, e.Attribute(anom, mk(1, 2))); got != CauseContention {
		t.Fatalf("1->2 procs: got %s", got)
	}
	if got := top(t, e.Attribute(anom, mk(1, 1))); got != CauseUnknown {
		t.Fatalf("unchanged count must not fire: %s", got)
	}
	// PIDs hidden (0 = unknown): stay silent instead of guessing.
	if got := top(t, e.Attribute(anom, mk(0, 0))); got != CauseUnknown {
		t.Fatalf("hidden PIDs must not fire: %s", got)
	}
	// Needs a baseline.
	only := []types.Event{gpu(105*sec, procs(3))}
	if got := top(t, e.Attribute(anom, only)); got != CauseUnknown {
		t.Fatalf("no baseline must not fire: %s", got)
	}
}
