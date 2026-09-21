package coordinator

import (
	"testing"

	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/attribution"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/types"
)

// trace builds a 4-rank run of 200 collectives, 20ms apart. From collective 100
// rank 1 is throttled and arrives 3ms late.
func trace(faulty bool) []types.Event {
	var evs []types.Event
	for s := 0; s < 200; s++ {
		ts := int64(s) * 20e6
		late := faulty && s >= 100
		for r := uint32(0); r < 4; r++ {
			d := int64(10e6) + int64(s%3)*1e4 // tiny deterministic jitter
			if late && r != 1 {
				d += 3e6
			}
			evs = append(evs, types.NCCLCollectiveEvent{
				BaseEvent: types.BaseEvent{Timestamp: ts, RankID: r}, PGID: "pg0", SeqID: uint64(s), OpType: "AllReduce", DurationNs: d})
		}
		if s%10 == 0 {
			m := types.GPUMetricEvent{BaseEvent: types.BaseEvent{Timestamp: ts, RankID: 1}, Utilization: 90}
			if late {
				m.ThrottleReasons = 0x40
			}
			evs = append(evs, m)
		}
	}
	return evs
}

func TestEndToEndThrottledRank(t *testing.T) {
	rep := Analyze(trace(true), Options{})
	if len(rep.Attributions) != 1 {
		t.Fatalf("want 1 attribution, got %d", len(rep.Attributions))
	}
	a := rep.Attributions[0]
	if a.Anomaly.StragglerRank != 1 || a.CausalChain[0].Cause != attribution.CauseThermalThrottle {
		t.Fatalf("wrong result: %s", a.Summary)
	}
}

func TestEndToEndCleanRunIsQuiet(t *testing.T) {
	if rep := Analyze(trace(false), Options{}); len(rep.Attributions) != 0 {
		t.Fatalf("false positive: %s", rep.Attributions[0].Summary)
	}
}
