package sim

import (
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/attribution"
	"github.com/RyanJHamby/distributed-gpu-training-flight-recorder/internal/coordinator"
)

// Outcome scores one run.
type Outcome struct {
	Detected        bool // the true straggler was flagged (clean: nothing flagged)
	FalseStragglers int  // flagged ranks that are not the true straggler
	AttrCorrect     bool // the true straggler's chain contains every true cause, ranked in the top len(causes)
	ExactTop        bool // the top cause is one of the true causes
}

func Score(p Params) Outcome {
	evs, truth := Generate(p)
	rep := coordinator.Analyze(evs, coordinator.Options{})
	var o Outcome
	if truth.Straggler < 0 {
		o.Detected = len(rep.Attributions) == 0
		o.FalseStragglers = len(rep.Attributions)
		return o
	}
	for _, a := range rep.Attributions {
		if int(a.Anomaly.StragglerRank) != truth.Straggler {
			o.FalseStragglers++
			continue
		}
		o.Detected = true
		want := map[attribution.CauseType]bool{}
		for _, c := range truth.Causes {
			want[c] = true
		}
		got := map[attribution.CauseType]bool{}
		for i, l := range a.CausalChain {
			if i < len(truth.Causes) {
				got[l.Cause] = true
			}
		}
		o.AttrCorrect = len(got) == len(want)
		for c := range want {
			o.AttrCorrect = o.AttrCorrect && got[c]
		}
		o.ExactTop = want[a.CausalChain[0].Cause]
	}
	return o
}
