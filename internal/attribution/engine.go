package attribution

import (
	"fmt"
	"sort"
	"strings"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/correlator"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

type CauseType string

const (
	CauseThermalThrottle CauseType = "thermal_throttle"
	CausePowerThrottle   CauseType = "power_throttle"
	CauseECCErrors       CauseType = "ecc_errors"
	CausePCIeBandwidth   CauseType = "pcie_bandwidth_drop"
	CauseNVLinkDegraded  CauseType = "nvlink_degraded"
	CauseMemoryPressure  CauseType = "memory_pressure"
	CauseClockReduced    CauseType = "clock_reduced"
	CauseContention      CauseType = "gpu_contention"
	CauseHostStall       CauseType = "host_stall"
	CauseUnknown         CauseType = "unknown"
)

// NVML clocksThrottleReasons bits.
const (
	throttleSWPowerCap    = 0x4
	throttleHWSlowdown    = 0x8
	throttleSWThermal     = 0x20
	throttleHWThermal     = 0x40
	throttleHWPowerBrake  = 0x80
	thermalMask           = throttleSWThermal | throttleHWThermal | throttleHWSlowdown
	powerMask             = throttleSWPowerCap | throttleHWPowerBrake
	bandwidthDropFraction = 0.5  // observed < 50% of baseline
	memPressurePct        = 90.0 // mean memory-bandwidth utilisation
	hostStallUtilPct      = 40.0 // mean GPU utilisation
	clockDropFraction     = 0.75 // SM clock < 75% of baseline
)

type CausalLink struct {
	Cause      CauseType   `json:"cause"`
	Evidence   types.Event `json:"-"`
	Confidence float64     `json:"confidence"`
	Detail     string      `json:"detail"`
}

type Attribution struct {
	Anomaly     correlator.Anomaly `json:"anomaly"`
	CausalChain []CausalLink       `json:"causal_chain"`
	RuledOut    []CauseType        `json:"ruled_out,omitempty"`
	Summary     string             `json:"summary"`
}

type Engine struct {
	attributionWindowNs int64
}

func NewEngine(attributionWindowNs int64) *Engine {
	return &Engine{attributionWindowNs: attributionWindowNs}
}

// Attribute explains an anomaly from the straggler rank's hardware events.
// "During" is [StartNs-window, EndNs]; anything earlier is the baseline that
// bandwidth rules compare against. Rules never fire on absent data: a missing
// signal is reported as ruled-out only when the collector produced samples.
func (e *Engine) Attribute(anomaly correlator.Anomaly, rankEvents []types.Event) Attribution {
	lo := anomaly.StartNs - e.attributionWindowNs
	var during, before []types.Event
	for _, ev := range rankEvents {
		ts := ev.TimestampNs()
		switch {
		case ts >= lo && ts <= anomaly.EndNs:
			during = append(during, ev)
		case ts < lo:
			before = append(before, ev)
		}
	}

	res := Attribution{Anomaly: anomaly}
	rules := []func([]types.Event, []types.Event) (CausalLink, bool, bool){
		ruleThermal, rulePower, ruleClock, ruleContention, ruleECC, rulePCIe, ruleNVLink, ruleMemory, ruleHostStall,
	}
	names := []CauseType{CauseThermalThrottle, CausePowerThrottle, CauseClockReduced, CauseContention, CauseECCErrors,
		CausePCIeBandwidth, CauseNVLinkDegraded, CauseMemoryPressure, CauseHostStall}
	for i, r := range rules {
		link, fired, hadData := r(during, before)
		switch {
		case fired:
			res.CausalChain = append(res.CausalChain, link)
		case hadData:
			res.RuledOut = append(res.RuledOut, names[i])
		}
	}
	sort.SliceStable(res.CausalChain, func(i, j int) bool {
		return res.CausalChain[i].Confidence > res.CausalChain[j].Confidence
	})
	if len(res.CausalChain) == 0 {
		res.CausalChain = []CausalLink{{Cause: CauseUnknown, Confidence: 0,
			Detail: "no hardware signal explains the lag"}}
	}
	res.Summary = summarize(anomaly, res)
	return res
}

func summarize(a correlator.Anomaly, r Attribution) string {
	top := r.CausalChain[0]
	s := fmt.Sprintf("rank %d straggled on %s (%d/%d collectives, ~%.1fms lag): %s (confidence %.2f) - %s",
		a.StragglerRank, a.OpType, a.Hits, a.Groups, float64(a.LagNs)/1e6,
		top.Cause, top.Confidence, top.Detail)
	if len(r.CausalChain) > 1 && top.Cause != CauseUnknown {
		var also []string
		for _, l := range r.CausalChain[1:] {
			also = append(also, string(l.Cause))
		}
		s += "; also present: " + strings.Join(also, ", ")
	}
	return s
}

// ruleThermal / rulePower use the NVML throttle bitmask (GPUMetricEvent) or
// ThermalEvent.ThrottleActive. Confidence is the fraction of samples throttled.
func ruleThermal(during, _ []types.Event) (CausalLink, bool, bool) {
	var n, hit int
	var ev types.Event
	for _, e := range during {
		switch m := e.(type) {
		case types.GPUMetricEvent:
			n++
			if m.ThrottleReasons&thermalMask != 0 {
				hit++
				ev = e
			}
		case types.ThermalEvent:
			n++
			if m.ThrottleActive {
				hit++
				ev = e
			}
		}
	}
	if n == 0 {
		return CausalLink{}, false, false
	}
	if hit == 0 {
		return CausalLink{}, false, true
	}
	return CausalLink{CauseThermalThrottle, ev, frac(hit, n),
		fmt.Sprintf("thermal throttle active in %d/%d samples", hit, n)}, true, true
}

func rulePower(during, _ []types.Event) (CausalLink, bool, bool) {
	var n, hit int
	var ev types.Event
	for _, e := range during {
		if m, ok := e.(types.GPUMetricEvent); ok {
			n++
			if m.ThrottleReasons&powerMask != 0 {
				hit++
				ev = e
			}
		}
	}
	if n == 0 {
		return CausalLink{}, false, false
	}
	if hit == 0 {
		return CausalLink{}, false, true
	}
	return CausalLink{CausePowerThrottle, ev, frac(hit, n),
		fmt.Sprintf("power cap/brake active in %d/%d samples", hit, n)}, true, true
}

// ruleClock catches slowdowns that carry no throttle-reason bit (locked or
// application clocks, or a throttle the driver does not report). It needs a
// pre-window baseline and stays below the explicit throttle rules in confidence.
func ruleClock(during, before []types.Event) (CausalLink, bool, bool) {
	clocks := func(evs []types.Event) (v []float64, last types.Event) {
		for _, e := range evs {
			if m, ok := e.(types.GPUMetricEvent); ok && m.SMClockMHz > 0 {
				v, last = append(v, m.SMClockMHz), e
			}
		}
		return
	}
	d, ev := clocks(during)
	b, _ := clocks(before)
	if len(d) == 0 || len(b) == 0 {
		return CausalLink{}, false, false
	}
	dm, bm := correlator.ComputeStats(d).Median, correlator.ComputeStats(b).Median
	if dm >= bm*clockDropFraction {
		return CausalLink{}, false, true
	}
	drop := 1 - dm/bm
	return CausalLink{CauseClockReduced, ev, min(0.4+drop/2, 0.7),
		fmt.Sprintf("SM clock fell %.0f%% vs baseline (%.0f -> %.0f MHz)", drop*100, bm, dm)}, true, true
}

// ruleContention: another process started using the GPU. Needs a baseline
// process count (>0) from before the window; a count of 0 means NVML could not
// tell (unsupported or PIDs hidden), so the rule stays silent rather than guess.
func ruleContention(during, before []types.Event) (CausalLink, bool, bool) {
	procs := func(evs []types.Event) (v []float64, last types.Event) {
		for _, e := range evs {
			if m, ok := e.(types.GPUMetricEvent); ok && m.ComputeProcs > 0 {
				v, last = append(v, float64(m.ComputeProcs)), e
			}
		}
		return
	}
	d, ev := procs(during)
	b, _ := procs(before)
	if len(d) == 0 || len(b) == 0 {
		return CausalLink{}, false, false
	}
	dm, bm := correlator.ComputeStats(d).Median, correlator.ComputeStats(b).Median
	if dm <= bm {
		return CausalLink{}, false, true
	}
	return CausalLink{CauseContention, ev, 0.7,
		fmt.Sprintf("compute processes on this GPU rose from %.0f to %.0f", bm, dm)}, true, true
}

// ruleECC fires on any counter increase across the window; DBE is stronger.
func ruleECC(during, before []types.Event) (CausalLink, bool, bool) {
	var first, last *types.GPUMetricEvent
	seq := append(append([]types.Event{}, lastOf(before)...), during...)
	for _, e := range seq {
		if m, ok := e.(types.GPUMetricEvent); ok {
			m := m
			if first == nil {
				first = &m
			}
			last = &m
		}
	}
	if first == nil {
		return CausalLink{}, false, false
	}
	dS := int64(last.ECCErrorsSBE) - int64(first.ECCErrorsSBE)
	dD := int64(last.ECCErrorsDBE) - int64(first.ECCErrorsDBE)
	if dS <= 0 && dD <= 0 {
		return CausalLink{}, false, true
	}
	conf := 0.6
	if dD > 0 {
		conf = 0.9
	}
	return CausalLink{CauseECCErrors, *last, conf,
		fmt.Sprintf("ECC counters rose: +%d single-bit, +%d double-bit", max(dS, 0), max(dD, 0))}, true, true
}

func rulePCIe(during, before []types.Event) (CausalLink, bool, bool) {
	var d, b []float64
	var ev types.Event
	for _, e := range during {
		if m, ok := e.(types.PCIeBandwidthEvent); ok {
			d = append(d, m.ReadBandwidthMBs+m.WriteBandwidthMBs)
			ev = e
		}
	}
	for _, e := range before {
		if m, ok := e.(types.PCIeBandwidthEvent); ok {
			b = append(b, m.ReadBandwidthMBs+m.WriteBandwidthMBs)
		}
	}
	return bandwidthDrop(CausePCIeBandwidth, "PCIe", d, b, ev)
}

func ruleNVLink(during, before []types.Event) (CausalLink, bool, bool) {
	var d, b []float64
	var ev types.Event
	for _, e := range during {
		if m, ok := e.(types.NVLinkEvent); ok {
			d = append(d, m.ThroughputGB)
			ev = e
		}
	}
	for _, e := range before {
		if m, ok := e.(types.NVLinkEvent); ok {
			b = append(b, m.ThroughputGB)
		}
	}
	return bandwidthDrop(CauseNVLinkDegraded, "NVLink", d, b, ev)
}

// bandwidthDrop needs a baseline: without pre-window samples there is nothing
// to compare to, so it neither fires nor claims to have ruled the cause out.
func bandwidthDrop(c CauseType, name string, d, b []float64, ev types.Event) (CausalLink, bool, bool) {
	if len(d) == 0 || len(b) == 0 {
		return CausalLink{}, false, false
	}
	dm, bm := correlator.ComputeStats(d).Median, correlator.ComputeStats(b).Median
	if bm <= 0 || dm >= bm*bandwidthDropFraction {
		return CausalLink{}, false, true
	}
	drop := 1 - dm/bm
	return CausalLink{c, ev, min(0.5+drop/2, 0.95),
		fmt.Sprintf("%s throughput fell %.0f%% vs baseline (%.1f -> %.1f)", name, drop*100, bm, dm)}, true, true
}

func ruleMemory(during, _ []types.Event) (CausalLink, bool, bool) {
	var v []float64
	var ev types.Event
	for _, e := range during {
		if m, ok := e.(types.GPUMetricEvent); ok {
			v = append(v, m.MemBandwidth)
			ev = e
		}
	}
	if len(v) == 0 {
		return CausalLink{}, false, false
	}
	mean := correlator.ComputeStats(v).Mean
	if mean < memPressurePct {
		return CausalLink{}, false, true
	}
	return CausalLink{CauseMemoryPressure, ev, 0.5,
		fmt.Sprintf("memory bandwidth saturated (mean %.0f%%)", mean)}, true, true
}

// ruleHostStall: a straggling rank whose GPU is mostly idle was waiting on the
// host (dataloader, CPU, Python), not on the device. Suppressed when a hardware
// throttle is present, since a throttled-then-idle GPU is a device problem.
func ruleHostStall(during, _ []types.Event) (CausalLink, bool, bool) {
	var v []float64
	var ev types.Event
	for _, e := range during {
		if m, ok := e.(types.GPUMetricEvent); ok {
			if m.ThrottleReasons&(thermalMask|powerMask) != 0 {
				return CausalLink{}, false, true
			}
			v = append(v, m.Utilization)
			ev = e
		}
	}
	if len(v) == 0 {
		return CausalLink{}, false, false
	}
	mean := correlator.ComputeStats(v).Mean
	if mean >= hostStallUtilPct {
		return CausalLink{}, false, true
	}
	return CausalLink{CauseHostStall, ev, min(0.4+(hostStallUtilPct-mean)/100, 0.8),
		fmt.Sprintf("GPU mostly idle (mean util %.0f%%) while rank lagged", mean)}, true, true
}

// lastOf returns the final GPUMetricEvent of evs as a one-element slice, giving
// counter deltas a starting point just before the window.
func lastOf(evs []types.Event) []types.Event {
	for i := len(evs) - 1; i >= 0; i-- {
		if _, ok := evs[i].(types.GPUMetricEvent); ok {
			return []types.Event{evs[i]}
		}
	}
	return nil
}

// frac caps at 0.95: sample-fraction evidence is never certainty.
func frac(a, b int) float64 { return min(float64(a)/float64(b), 0.95) }
