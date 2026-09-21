package sim

import "testing"

func TestGenerateIsDeterministic(t *testing.T) {
	a, ta := Generate(Params{Fault: Thermal, Seed: 7})
	b, tb := Generate(Params{Fault: Thermal, Seed: 7})
	if len(a) != len(b) || ta.Straggler != tb.Straggler {
		t.Fatal("same seed must give same trace")
	}
	if _, tc := Generate(Params{Fault: Clean, Seed: 7}); tc.Straggler != -1 {
		t.Fatal("clean run must have no straggler")
	}
}

// Regression floor at default noise: each fault type must be found and
// attributed correctly in nearly every seed. Failures here mean a real change
// in detector or rule behaviour.
func TestScorecardFloor(t *testing.T) {
	const seeds = 20
	for _, f := range AllFaults {
		var det, attr, fp int
		for s := int64(0); s < seeds; s++ {
			o := Score(Params{Fault: f, Seed: s})
			if o.Detected {
				det++
			}
			if o.AttrCorrect {
				attr++
			}
			fp += o.FalseStragglers
		}
		if det < seeds-1 || fp > 0 {
			t.Errorf("%s: detected %d/%d, false stragglers %d", f, det, seeds, fp)
		}
		if f != Clean && attr < seeds-1 {
			t.Errorf("%s: attributed %d/%d", f, attr, seeds)
		}
	}
}
