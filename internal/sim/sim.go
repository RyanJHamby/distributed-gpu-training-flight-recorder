// Package sim generates synthetic multi-rank training traces with a known
// injected fault, so detection and attribution can be scored against ground
// truth. Synthetic results validate the logic only, not real-hardware behaviour.
package sim

import (
	"math/rand"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/attribution"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

type Fault string

const (
	Clean      Fault = "clean"
	Thermal    Fault = "thermal"
	Power      Fault = "power"
	ECC        Fault = "ecc"
	PCIe       Fault = "pcie"
	NVLink     Fault = "nvlink"
	HostStall  Fault = "host_stall"
	Contention Fault = "contention" // a second process shares the GPU (visible via NVML process count)
	// ContentionHidden is the same fault where NVML cannot see other processes' PIDs
	// (typical inside containers): the tool must say "unknown", not guess.
	ContentionHidden Fault = "contention_hidden"
	Compound         Fault = "thermal+ecc"
)

var AllFaults = []Fault{Clean, Thermal, Power, ECC, PCIe, NVLink, HostStall, Contention, ContentionHidden, Compound}

// Params controls one synthetic run.
type Params struct {
	Fault    Fault
	Seed     int64
	Ranks    int     // default 8
	Steps    int     // default 600
	Jitter   float64 // relative std-dev of op duration, default 0.03
	Severity float64 // straggler lag as a fraction of op duration, default 0.3
}

// Truth is the ground truth for a run.
type Truth struct {
	Fault     Fault
	Straggler int // -1 when clean
	Causes    []attribution.CauseType
}

const (
	stepNs   = int64(100e6) // 100ms per step
	opNs     = float64(10e6)
	onsetDiv = 2 // fault starts halfway through the run
)

// Generate builds a trace. Every rank emits one collective per step and a
// hardware sample every 5 steps.
func Generate(p Params) ([]types.Event, Truth) {
	if p.Ranks == 0 {
		p.Ranks = 8
	}
	if p.Steps == 0 {
		p.Steps = 600
	}
	if p.Jitter == 0 {
		p.Jitter = 0.03
	}
	if p.Severity == 0 {
		p.Severity = 0.3
	}
	rng := rand.New(rand.NewSource(p.Seed))
	truth := Truth{Fault: p.Fault, Straggler: -1}
	if p.Fault != Clean {
		truth.Straggler = rng.Intn(p.Ranks)
		truth.Causes = causesFor(p.Fault)
	}
	onset := p.Steps / onsetDiv
	var evs []types.Event
	var eccS, eccD uint64

	for s := 0; s < p.Steps; s++ {
		ts := int64(s) * stepNs
		faulty := p.Fault != Clean && s >= onset
		for r := 0; r < p.Ranks; r++ {
			dur := opNs * (1 + rng.NormFloat64()*p.Jitter)
			if faulty && r != truth.Straggler {
				dur += opNs * p.Severity // peers wait for the straggler
			}
			base := types.BaseEvent{Timestamp: ts, RankID: uint32(r), NodeID: "sim"}
			evs = append(evs, types.NCCLCollectiveEvent{BaseEvent: base, PGID: "pg0",
				SeqID: uint64(s), OpType: "AllReduce", DataSize: 1 << 28, DurationNs: int64(dur)})

			if s%5 != 0 {
				continue
			}
			isS := faulty && r == truth.Straggler
			baseProcs := 1
			if p.Fault == ContentionHidden {
				baseProcs = 0 // unknown
			}
			m := types.GPUMetricEvent{BaseEvent: base, Temperature: 65 + rng.Float64()*3,
				PowerWatts: 300, Utilization: 92 + rng.Float64()*4, MemBandwidth: 55 + rng.Float64()*10,
				SMClockMHz: 1900, ComputeProcs: baseProcs}
			pcie := 12000 * (1 + rng.NormFloat64()*0.03)
			nvl := 200 * (1 + rng.NormFloat64()*0.03)
			if isS {
				switch p.Fault {
				case Thermal, Compound:
					m.ThrottleReasons, m.Temperature, m.SMClockMHz = 0x40, 88, 1100
				case Power:
					m.ThrottleReasons, m.SMClockMHz = 0x4, 1300
				case Contention:
					m.ComputeProcs = 2 // a second process shares this GPU
				case HostStall:
					m.Utilization = 10 + rng.Float64()*10
				case PCIe:
					pcie *= 0.25
				case NVLink:
					nvl *= 0.2
				}
				if p.Fault == ECC || p.Fault == Compound {
					eccS += 3
					eccD++
				}
			}
			m.ECCErrorsSBE, m.ECCErrorsDBE = eccS, eccD
			evs = append(evs, m,
				types.PCIeBandwidthEvent{BaseEvent: base, ReadBandwidthMBs: pcie},
				types.NVLinkEvent{BaseEvent: base, LinkID: 0, ThroughputGB: nvl})
		}
	}
	return evs, truth
}

func causesFor(f Fault) []attribution.CauseType {
	switch f {
	case Thermal:
		return []attribution.CauseType{attribution.CauseThermalThrottle}
	case Power:
		return []attribution.CauseType{attribution.CausePowerThrottle}
	case ECC:
		return []attribution.CauseType{attribution.CauseECCErrors}
	case PCIe:
		return []attribution.CauseType{attribution.CausePCIeBandwidth}
	case NVLink:
		return []attribution.CauseType{attribution.CauseNVLinkDegraded}
	case HostStall:
		return []attribution.CauseType{attribution.CauseHostStall}
	case Compound:
		return []attribution.CauseType{attribution.CauseThermalThrottle, attribution.CauseECCErrors}
	case Contention:
		return []attribution.CauseType{attribution.CauseContention}
	case ContentionHidden:
		return []attribution.CauseType{attribution.CauseUnknown} // located, not explained
	}
	return nil
}
